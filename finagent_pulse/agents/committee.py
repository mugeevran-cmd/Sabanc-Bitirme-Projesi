"""The autonomous investment committee: three agents wired with LangGraph.

    Data Analyst  ->  Sentiment Critic  ->  Risk Manager  ->  directive

Each node computes a structured ``findings`` dict deterministically and then
asks the narrative layer to explain it (see ``agents.llm`` for why the decision
and the prose are kept separate).  State flows forward, so the Risk Manager
sees both upstream analyses and can act on their *disagreement*, which is the
central idea of the design: a bull or bear trap is exactly the situation where
the quantitative model and the news sentiment point in opposite directions.
"""
from __future__ import annotations

import logging
import pathlib
import re
from typing import Annotated, Any, TypedDict

import numpy as np
import pandas as pd

from finagent_pulse import config
from finagent_pulse.agents.llm import get_writer
from finagent_pulse.models.forecaster import ForecastService
from finagent_pulse.models.sentiment import fear_greed_at
from finagent_pulse.rag.hybrid import get_retriever

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------
def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class CommitteeState(TypedDict, total=False):
    as_of: str
    features: Any                 # the feature table (passed by reference)
    narrative: bool               # False -> deterministic templates, no model calls
    quant: Annotated[dict, _merge]
    sentiment: Annotated[dict, _merge]
    risk: Annotated[dict, _merge]
    reports: Annotated[dict, _merge]
    # Narration routed through files instead of an API. See ``_narrate``.
    brief_dir: str | None
    narration_dir: str | None


# --------------------------------------------------------------------------
# Narration
# --------------------------------------------------------------------------
# The agents already compute *why* they decided what they decided -- the rule
# that fired, how close the call came to being a different one, the principle
# that governs it, what would void it.  These helpers hand that chain to the
# model as labelled sections rather than as a stringified Python dict, so the
# prose can be a justification instead of a paraphrase.

def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    if isinstance(v, list) and v and isinstance(v[0], float):
        return "[" + ", ".join(f"{x:,.2f}" for x in v) + "]"
    return str(v)


def _fmt_fields(f: dict, skip: tuple[str, ...] = ()) -> str:
    """Scalar findings as ``key: value`` lines.

    The narrative fields (``reasons``, ``observations``, ``principles``) are
    always skipped: they are prose already and get their own sections, where
    the model can see them as the argument they are rather than as a nested
    repr buried three levels inside a dict.
    """
    drop = set(skip) | {"reasons", "observations", "principles", "evidence"}
    return "\n".join(f"- {k}: {_fmt_value(v)}"
                     for k, v in f.items() if k not in drop) or "- (none)"


def _fmt_observations(obs: list[dict] | None) -> str:
    """Cross-checks, integrity flags first -- they outrank everything else."""
    if not obs:
        return "- (none fired: the numbers cross-check cleanly)"
    order = sorted(obs, key=lambda o: o["severity"] != "integrity")
    return "\n".join(f"- [{o['severity']}] {o['text']}" for o in order)


def _fmt_evidence(evidence: list[dict] | None, limit: int = 8) -> str:
    """Retrieved headlines, with the retrieval arms that surfaced each one."""
    if not evidence:
        return "- (nothing retrieved for this window)"
    return "\n".join(
        f"- [{d['date']}] ({d['label']}, {d['sentiment']:+.2f}) via {d['provenance']}"
        f" -- {d['headline']}"
        for d in evidence[:limit])


def _brief(*sections: tuple[str, str]) -> str:
    """Assemble the labelled sections a narration prompt is built from."""
    return "\n\n".join(f"## {title}\n{body}" for title, body in sections if body)


_ANCHOR = (
    "## The report the rules already produced\n"
    "Below is the deterministic rendering of everything above. Your prose must "
    "carry every claim it makes, in the same order of importance. Do not "
    "introduce a number, date, ticker or headline that does not appear in this "
    "brief -- if something you would normally comment on is missing, say it is "
    "not available rather than supplying it.\n\n"
)


def _write_brief(directory: str, agent: str, system: str, prompt: str) -> None:
    """Write exactly what a model would be sent, for narration done elsewhere.

    The brief is the whole interface to the narrative layer: everything the
    prose is allowed to contain is in it. Writing it out means the narration can
    be produced by the Anthropic API, by another provider, or by a person in an
    assistant session, without any of them needing to run the pipeline.
    """
    out = pathlib.Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{agent}.brief.md").write_text(
        f"<!-- FinAgent-Pulse narration brief for the {agent}. Write the prose "
        f"this asks for and save it as {agent}.md in the narration directory. -->\n\n"
        f"## System prompt\n{system}\n\n{prompt}\n")


def _read_narration(directory: str, agent: str, fallback: str) -> str:
    """Read prose written against an exported brief; template if absent."""
    path = pathlib.Path(directory) / f"{agent}.md"
    if not path.exists():
        log.warning("no narration at %s; using the deterministic report", path)
        return fallback
    text = path.read_text().strip()
    if not text:
        log.warning("narration at %s is empty; using the deterministic report", path)
        return fallback
    return text


def narration_mode(state: CommitteeState | dict) -> str:
    """How the prose in this state was produced -- for the report footer."""
    if not state.get("narrative", True):
        return "template"
    if state.get("narration_dir"):
        return "assisted"
    return get_writer().mode


def _narrate(state: CommitteeState, agent: str, system: str, prompt: str,
             fallback: str) -> str:
    """Write agent prose, unless the caller asked for findings only.

    The backtest runs the committee once per decision and throws every word of
    prose away. With an API key configured that was 3 model calls x 75 decisions
    = 225 requests for output nobody reads, and it made the backtest depend on a
    network round-trip. Callers that only want the findings pass
    ``narrative=False`` and get the deterministic template instead.

    The deterministic report is appended to every prompt as an anchor. It costs
    a few hundred tokens and buys two things: the model writes *from* the
    committee's own reasoning instead of reconstructing it from raw fields, and
    every figure it could legitimately cite is already on the page, so there is
    nothing it needs to invent to finish a sentence.
    """
    if not state.get("narrative", True):
        return fallback
    full = f"{prompt}\n\n{_ANCHOR}{fallback}"
    if state.get("brief_dir"):
        _write_brief(state["brief_dir"], agent, system, full)
    if state.get("narration_dir"):
        return _read_narration(state["narration_dir"], agent, fallback)
    return get_writer().write(system, full, fallback)


# --------------------------------------------------------------------------
# Agent 1 -- Data Analyst
# --------------------------------------------------------------------------
DATA_ANALYST_SYSTEM = (
    "You are a quantitative data analyst on an institutional investment "
    "committee. You interpret model output and price structure. You never "
    "invent numbers: every figure you cite must appear in the findings given "
    "to you. Write 3-5 tight paragraphs of markdown, no preamble."
)

# A numbered skeleton, rather than "explain the findings". The committee's
# value is in the order it reasons: magnitude is meaningless until it has been
# measured against noise, and none of it is worth reading if the cross-checks
# say the inputs are broken.
QUANT_TASK = (
    "Work through, in this order:\n"
    "1. What the model projects, and the direction of the trajectory.\n"
    "2. Whether that move is large enough to trade against 7-day noise -- lead "
    "with the signal-to-noise ratio against the conviction floor, not with the "
    "raw percentage.\n"
    "3. What the price structure says: volatility regime, RSI, distance from "
    "the 50-day average.\n"
    "4. Every cross-check that fired, and what it means for how much weight the "
    "forecast can carry.\n"
    "If any cross-check is marked `integrity`, open with it: a forecast built "
    "on inputs the committee doubts is not a forecast, and no amount of "
    "interpretation downstream repairs that."
)


def data_analyst_node(state: CommitteeState) -> dict:
    df: pd.DataFrame = state["features"]
    as_of = pd.Timestamp(state["as_of"])
    hist = df[df["date"] <= as_of]
    row = hist.iloc[-1]

    forecast = _forecast_service().forecast(df, as_of=as_of)

    # Structural anomaly detection against the trailing distribution.
    trailing = hist.tail(250)
    ret_z = float((row["log_return"] - trailing["log_return"].mean())
                  / (trailing["log_return"].std() + 1e-9))
    vol_pct = float((trailing["volatility_20d"] <= row["volatility_20d"]).mean())
    vol_regime = ("high" if vol_pct >= config.HIGH_VOL_PERCENTILE
                  else "low" if vol_pct <= 0.20 else "normal")

    horizon_vol = float(row["volatility_20d"] * np.sqrt(config.LSTM.horizon))
    expected = forecast["total_return_pct"] / 100.0
    # Is the forecast large enough to be tradable against 7-day noise?
    signal_to_noise = float(abs(expected) / (horizon_vol + 1e-9))

    anomalies = []
    if abs(ret_z) >= 2.0:
        anomalies.append(f"latest session moved {ret_z:+.1f} sigma versus its 1-year distribution")
    if vol_regime == "high":
        anomalies.append(f"20-day volatility sits in the {vol_pct:.0%} percentile of the past year")
    if row["rsi_14"] >= 70:
        anomalies.append(f"RSI(14) at {row['rsi_14']:.0f} indicates overbought conditions")
    elif row["rsi_14"] <= 30:
        anomalies.append(f"RSI(14) at {row['rsi_14']:.0f} indicates oversold conditions")
    if abs(row["dist_ma50"]) >= 0.05:
        anomalies.append(f"price is {row['dist_ma50']:+.1%} away from its 50-day average")
    if float(row["headline_count_z"]) >= 2.0:
        anomalies.append(f"news coverage is {row['headline_count_z']:.1f} sigma above normal")

    findings = {
        "as_of": str(as_of.date()),
        "close": float(row["close"]),
        "forecast_7d_pct": forecast["total_return_pct"],
        "forecast_path_pct": forecast["path_return_pct"],
        "forecast_prices": forecast["prices"],
        "trend_direction": "up" if expected > 0 else "down",
        "return_zscore": ret_z,
        "volatility_20d": float(row["volatility_20d"]),
        "volatility_percentile": vol_pct,
        "volatility_regime": vol_regime,
        "horizon_volatility_pct": horizon_vol * 100,
        "signal_to_noise": signal_to_noise,
        "rsi_14": float(row["rsi_14"]),
        "dist_ma50_pct": float(row["dist_ma50"]) * 100,
        "anomalies": anomalies,
    }
    findings["observations"] = _quant_observations(findings, hist)

    fallback = _render_quant_report(findings)
    prose = _narrate(
        state,
        "data_analyst",
        DATA_ANALYST_SYSTEM,
        _brief(
            ("What to write", QUANT_TASK),
            ("Measurements", _fmt_fields(findings, skip=("forecast_prices",))),
            ("Structural anomalies",
             "\n".join(f"- {a}" for a in findings["anomalies"])
             or "- (none: price action is within its trailing distribution)"),
            ("Cross-checks on the forecast itself",
             _fmt_observations(findings["observations"])),
        ),
        fallback,
    )
    return {"quant": findings, "reports": {"data_analyst": prose}}


def _quant_observations(f: dict, hist: pd.DataFrame) -> list[dict]:
    """Cross-checks the individual numbers cannot make on their own.

    The template used to relay each figure and stop, which left the reader to
    notice for themselves that a forecast was larger than anything the index has
    actually done, or that a "+0.6%" trajectory spends most of the horizon
    somewhere else. These are the checks a human analyst runs before quoting the
    model, computed rather than written, so they stay reproducible and testable.

    Each observation is ``{"kind", "severity", "text"}``. Severity is
    ``"integrity"`` when the numbers look wrong rather than merely unfavourable.
    """
    out: list[dict] = []
    expected = f["forecast_7d_pct"] / 100.0
    path = f["forecast_path_pct"]

    # 1. Is the projected move on the scale of moves this index actually makes?
    # Trailing 7-session returns, all strictly in the past of the decision date.
    trailing = np.log(hist["close"]).diff(config.LSTM.horizon).tail(250).dropna()
    if len(trailing) >= 60:
        pct = float((trailing.abs() <= abs(expected)).mean())
        if pct >= 0.99:
            out.append({
                "kind": "forecast_scale", "severity": "integrity",
                "text": (f"the projected {f['forecast_7d_pct']:+.2f}% is larger than "
                         f"{pct:.0%} of the 7-session moves this index actually made "
                         "in the trailing year. A forecast outside the realised "
                         "distribution is usually a data or scaling fault rather "
                         "than a signal, and should be reconciled before it is acted on"),
            })
        elif pct >= 0.90:
            out.append({
                "kind": "forecast_scale", "severity": "note",
                "text": (f"the projected move sits at the {pct:.0%} percentile of "
                         "trailing 7-session moves — large, but within what the "
                         "index does"),
            })

    # 2. Does the trajectory hold its move, or does it round-trip inside the week?
    peak = max(path, key=abs)
    if abs(peak) > 1e-9 and abs(path[-1]) < 0.5 * abs(peak):
        out.append({
            "kind": "path_shape", "severity": "note",
            "text": (f"the trajectory is not monotone: it reaches {peak:+.2f}% at "
                     f"t+{path.index(peak) + 1} and ends the week at "
                     f"{path[-1]:+.2f}%, so most of the projected move is given "
                     "back inside the horizon"),
        })

    # 3. Momentum and the forecast can point the same way and still disagree
    # about what to do -- buying an overbought tape is a different trade from
    # buying a base.
    if f["rsi_14"] >= 70 and expected > 0:
        out.append({
            "kind": "momentum_conflict", "severity": "note",
            "text": (f"the model projects further upside into an RSI of "
                     f"{f['rsi_14']:.0f}: this is a request to buy strength, not "
                     "a reversal setup"),
        })
    elif f["rsi_14"] <= 30 and expected < 0:
        out.append({
            "kind": "momentum_conflict", "severity": "note",
            "text": (f"the model projects further downside into an RSI of "
                     f"{f['rsi_14']:.0f}: this is a request to sell weakness, not "
                     "a reversal setup"),
        })

    # 4. An index trading far from its own 50-day average is either a genuine
    # dislocation or a mismatch between two series that should be on one scale.
    if abs(f["dist_ma50_pct"]) >= 20.0:
        out.append({
            "kind": "ma_dislocation", "severity": "integrity",
            "text": (f"price sits {f['dist_ma50_pct']:+.1f}% from its 50-day average. "
                     "For a broad index that is far outside normal dispersion and "
                     "points at stale or mis-scaled inputs rather than at a market move"),
        })
    return out


def _render_quant_report(f: dict) -> str:
    lines = [
        "### Data Analyst — Quantitative Assessment",
        "",
        f"As of **{f['as_of']}** the index closed at **{f['close']:,.2f}**. The Bi-LSTM "
        f"projects a **{f['forecast_7d_pct']:+.2f}%** move over the next 7 trading sessions, "
        f"a **{f['trend_direction']}ward** trajectory.",
        "",
        f"Expected 7-day volatility is **{f['horizon_volatility_pct']:.2f}%**, which places the "
        f"forecast's signal-to-noise ratio at **{f['signal_to_noise']:.2f}**. "
        + (f"That is below the {config.SIGNAL_TO_NOISE_MIN:.2f} conviction floor "
           "calibrated on validation data, so the directional call carries "
           "little actionable information on its own."
           if f["signal_to_noise"] < config.SIGNAL_TO_NOISE_MIN else
           f"That clears the {config.SIGNAL_TO_NOISE_MIN:.2f} conviction floor "
           "calibrated on validation data, placing this among the most "
           "informative days in the sample."),
        "",
        f"The market is in a **{f['volatility_regime']}-volatility** regime "
        f"({f['volatility_percentile']:.0%} percentile of the trailing year). "
        f"RSI(14) reads **{f['rsi_14']:.0f}** and price sits **{f['dist_ma50_pct']:+.1f}%** "
        "from its 50-day moving average.",
        "",
    ]
    if f["anomalies"]:
        lines.append("**Structural anomalies detected:**")
        lines += [f"- {a}" for a in f["anomalies"]]
    else:
        lines.append("**No structural anomalies detected** — price action is within "
                     "its normal trailing distribution.")

    obs = f.get("observations", [])
    integrity = [o for o in obs if o["severity"] == "integrity"]
    notes = [o for o in obs if o["severity"] != "integrity"]
    if integrity:
        lines += ["", "> **Data integrity — read the forecast with suspicion.** "
                  + " Also, ".join(o["text"] for o in integrity) + "."]
    if notes:
        lines += ["", "**Reading the forecast:** "
                  + "; ".join(o["text"] for o in notes) + "."]
    if not obs:
        lines += ["", "The forecast is internally consistent: its magnitude is "
                  "ordinary against the trailing distribution, the trajectory holds "
                  "its move across the horizon, and momentum does not contradict it."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Agent 2 -- Sentiment Critic
# --------------------------------------------------------------------------
SENTIMENT_CRITIC_SYSTEM = (
    "You are a news and sentiment analyst on an institutional investment "
    "committee. You read retrieved headlines critically and identify what is "
    "actually driving them, distinguishing a genuine information shift from "
    "crowd reaction. Cite only headlines present in the evidence. Write 3-5 "
    "tight paragraphs of markdown, no preamble."
)

SENTIMENT_TASK = (
    "Work through, in this order:\n"
    "1. Where the sentiment channel stands: the session reading, the 5-day "
    "average against its 20-day baseline, and the direction of travel.\n"
    "2. What the retrieval actually surfaced -- name the dominant drivers and "
    "quote at most three headlines from the evidence, by date.\n"
    "3. Whether this reads as genuine information or as crowd reaction. "
    "Coverage volume, one-sidedness and the Fear & Greed percentile are the "
    "evidence for that call.\n"
    "4. Every cross-check that fired, especially where the channel disagrees "
    "with itself.\n"
    "The retrieval diagnostics say how the evidence was found. If the knowledge "
    "graph expanded the query, say which concepts it reached for and what that "
    "surfaced -- that expansion is part of the reasoning, not plumbing."
)


def sentiment_critic_node(state: CommitteeState) -> dict:
    df: pd.DataFrame = state["features"]
    as_of = pd.Timestamp(state["as_of"])
    hist = df[df["date"] <= as_of]
    row = hist.iloc[-1]

    lookback_start = (as_of - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    query = ("What is driving the S&P 500 right now, and what are the main "
             "risks investors are reacting to?")
    docs, diag = get_retriever().retrieve(
        query, mode="hybrid_kg", top_k=config.RETRIEVAL_TOP_K,
        start=lookback_start, end=as_of.strftime("%Y-%m-%d"))

    recent = hist.tail(20)
    sent_now = float(row["sent_mean"])
    sent_5d = float(row["sent_mean_5d"])
    sent_baseline = float(recent["sent_mean"].mean())
    # Percentile of today's smoothed sentiment within the history available
    # on the decision date. `hist`, never `df`: ranking against the full
    # table would let sessions after `as_of` set the scale for this one.
    fg_index = fear_greed_at(hist)

    entity_counts: dict[str, int] = {}
    for d in docs:
        for e in d.entities:
            entity_counts[e] = entity_counts.get(e, 0) + 1
    drivers = sorted(entity_counts.items(), key=lambda kv: -kv[1])[:5]

    neg_share = float(row["sent_neg_ratio"])
    crowded = float(row["headline_count_z"]) >= 2.0

    if sent_now >= config.SENTIMENT_STRONG:
        stance = "bullish"
    elif sent_now <= -config.SENTIMENT_STRONG:
        stance = "bearish"
    else:
        stance = "neutral"

    contrarian = None
    if neg_share >= 0.70:
        contrarian = ("Negative headlines make up "
                      f"{neg_share:.0%} of coverage — one-sided pessimism is "
                      "historically a contrarian signal rather than a confirmation.")
    elif float(row["sent_pos_ratio"]) >= 0.70:
        contrarian = ("Positive headlines make up "
                      f"{row['sent_pos_ratio']:.0%} of coverage — uniformly "
                      "optimistic coverage often marks crowded positioning.")

    findings = {
        "as_of": str(as_of.date()),
        "sentiment_now": sent_now,
        "sentiment_5d": sent_5d,
        "sentiment_20d_baseline": sent_baseline,
        "sentiment_shift": sent_5d - sent_baseline,
        "fear_greed_index": fg_index,
        "stance": stance,
        "negative_share": neg_share,
        "positive_share": float(row["sent_pos_ratio"]),
        "headline_count": int(row["headline_count"]),
        "coverage_zscore": float(row["headline_count_z"]),
        "crowded_attention": crowded,
        "contrarian_flag": contrarian,
        "top_drivers": drivers,
        "retrieval": diag,
        "evidence": [d.as_dict() for d in docs],
    }
    findings["observations"] = _sentiment_observations(findings)

    evidence_block = "\n".join(
        f"- [{d.date}] ({d.label}, {d.sentiment:+.2f}) via {d.provenance} -- {d.headline}"
        for d in docs) or "- (nothing retrieved for this window)"
    fallback = _render_sentiment_report(findings, docs)
    prose = _narrate(
        state,
        "sentiment_critic",
        SENTIMENT_CRITIC_SYSTEM,
        _brief(
            ("What to write", SENTIMENT_TASK),
            ("Measurements", _fmt_fields(findings, skip=("retrieval",))),
            ("How the evidence was retrieved", _fmt_fields(diag)),
            ("Retrieved evidence", evidence_block),
            ("Cross-checks on the sentiment channel",
             _fmt_observations(findings["observations"])),
        ),
        fallback,
    )
    return {"sentiment": findings, "reports": {"sentiment_critic": prose}}


def _sentiment_observations(f: dict) -> list[dict]:
    """Where the sentiment channel disagrees with itself.

    The template reported four sentiment numbers side by side and left their
    disagreements unremarked -- most visibly a Fear & Greed reading in the greed
    band on a day whose own headlines are net negative, which happens whenever
    the tape turns, because the index is a 20-day smoothed percentile and today's
    score is not.
    """
    out: list[dict] = []
    now, fg = f["sentiment_now"], f["fear_greed_index"]

    # 1. A slow variable and a fast one, reported together, will disagree at
    # exactly the moments that matter.
    if fg >= 65 and now < 0:
        out.append({
            "kind": "regime_turn", "severity": "note",
            "text": (f"the Fear & Greed index reads {fg:.0f}/100 — the greed band — "
                     f"while today's headlines score {now:+.3f}, net negative. The "
                     "index is a 20-day smoothed percentile and today is not, so "
                     "this is what the start of a turn looks like rather than a "
                     "contradiction: positioning is still complacent, the news flow "
                     "has already rolled over"),
        })
    elif fg <= 35 and now > 0:
        out.append({
            "kind": "regime_turn", "severity": "note",
            "text": (f"the Fear & Greed index reads {fg:.0f}/100 — the fear band — "
                     f"while today's headlines score {now:+.3f}, net positive. "
                     "Sentiment is improving faster than the 20-day percentile can "
                     "register it"),
        })

    # 2. The retrieved evidence is the day's news as the RAG stack sees it. If
    # its mean disagrees in sign with the aggregate, the two views of "today's
    # news" are not the same view, and the quoted headlines below will read as
    # if they belong to a different session.
    ev = [d["sentiment"] for d in f.get("evidence", [])]
    if ev:
        ev_mean = sum(ev) / len(ev)
        if abs(now) >= 0.05 and abs(ev_mean) >= 0.05 and (ev_mean > 0) != (now > 0):
            out.append({
                "kind": "evidence_divergence", "severity": "note",
                "text": (f"the {len(ev)} headlines retrieved for context average "
                         f"{ev_mean:+.3f} against the session's {now:+.3f} — the "
                         "retrieved window spans 14 days, so it is carrying the "
                         "prior mood rather than today's"),
            })

    # 3. A mean over very few headlines is not a measurement.
    if f["headline_count"] <= 5:
        out.append({
            "kind": "thin_coverage", "severity": "integrity",
            "text": (f"the session's sentiment rests on {f['headline_count']} "
                     "headlines. That is too thin to average, and the reading "
                     "should not carry weight against the price channel"),
        })

    # 4. Direction of travel, which the raw levels do not show.
    shift = f["sentiment_shift"]
    if abs(shift) >= 0.10:
        out.append({
            "kind": "momentum", "severity": "note",
            "text": (f"the 5-day mean has moved {shift:+.3f} against its 20-day "
                     f"baseline, so the news environment is "
                     f"{'improving' if shift > 0 else 'deteriorating'} faster than "
                     "the level alone suggests"),
        })
    return out


def _render_sentiment_report(f: dict, docs) -> str:
    drivers = ", ".join(f"`{e}` ({n})" for e, n in f["top_drivers"]) or "no dominant entity"
    lines = [
        "### Sentiment Critic — News & Semantic Assessment",
        "",
        f"FinBERT scores the {f['headline_count']} headlines attached to "
        f"**{f['as_of']}** at a mean sentiment of **{f['sentiment_now']:+.3f}**, "
        f"a **{f['stance']}** reading. The 5-day average stands at "
        f"**{f['sentiment_5d']:+.3f}** against a 20-day baseline of "
        f"**{f['sentiment_20d_baseline']:+.3f}**, a shift of "
        f"**{f['sentiment_shift']:+.3f}**.",
        "",
        f"On the Market Fear & Greed scale — 20-day smoothed sentiment "
        f"percentile-ranked against everything known up to this date — this "
        f"sits at **{f['fear_greed_index']:.0f}/100**. "
        f"Coverage volume is **{f['coverage_zscore']:+.1f} sigma** versus its "
        "60-day norm"
        + (", an attention-driven regime in which prices reflect crowding as "
           "much as fundamentals." if f["crowded_attention"] else "."),
        "",
        f"Hybrid retrieval over the trailing 14 days surfaced "
        f"{len(f['evidence'])} headlines. The dominant drivers are {drivers}.",
    ]
    if f["retrieval"].get("kg_neighbours"):
        lines.append(
            f"The knowledge graph expanded the query toward "
            f"{', '.join('`' + n + '`' for n in f['retrieval']['kg_neighbours'])}, "
            "surfacing related themes the literal query would have missed.")
    lines += ["", "**Representative evidence:**"]
    lines += [f"- *[{d.date}]* ({d.label}, {d.sentiment:+.2f}) {d.headline}"
              for d in docs[:5]]
    if f["contrarian_flag"]:
        lines += ["", f"> **Contrarian note.** {f['contrarian_flag']}"]

    obs = f.get("observations", [])
    integrity = [o for o in obs if o["severity"] == "integrity"]
    notes = [o for o in obs if o["severity"] != "integrity"]
    if integrity:
        lines += ["", "> **Read this channel with suspicion.** "
                  + " Also, ".join(o["text"] for o in integrity) + "."]
    if notes:
        lines += ["", "**What the numbers disagree about:** "
                  + "; ".join(o["text"] for o in notes) + "."]
    if not obs:
        lines += ["", "The sentiment channel is internally consistent: the "
                  "smoothed index, the session score and the retrieved evidence "
                  "all point the same way."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Agent 3 -- Risk Manager
# --------------------------------------------------------------------------
RISK_MANAGER_SYSTEM = (
    "You are the risk manager on an institutional investment committee and you "
    "hold the final word. You are deliberately conservative: capital "
    "preservation outranks opportunity. You have been given a directive that "
    "was computed by rule; explain and justify it against the cited investment "
    "principles. Do not contradict the directive, and do not restate it as "
    "though it were your own judgement call -- it is the output of a rule, and "
    "your job is to show that the rule was right here. You never invent "
    "numbers or dates: every figure you cite must appear in the brief. Write "
    "3-5 tight paragraphs of markdown, no preamble."
)

# The directive is one word; the justification is the report. This is the shape
# a reader needs in order to audit the call rather than take it on faith --
# which rule fired, how close it came to firing the other way, what governs it,
# and what would prove it wrong.
RISK_TASK = (
    "Justify the computed directive, in this order:\n"
    "1. The rule that produced it. Name the gate and the numbers that cleared "
    "or failed it.\n"
    "2. How close this was to being a different call. The counterfactual is in "
    "the cross-checks -- a HOLD that missed the floor by a hair is a different "
    "report from one that was nowhere near, and the reader is entitled to know "
    "which one this is.\n"
    "3. The principle that governs the decision. Quote the retrieved principles "
    "and apply them to this specific state; do not cite one that has no "
    "bearing on what happened here.\n"
    "4. The invalidation frame: the horizon, the noise band, and what would "
    "void the thesis.\n"
    "If the cross-checks carry an `integrity` flag, that comes first and "
    "colours everything else: a directive resting on inputs the committee has "
    "already doubted must be reported as such, however sound the rule was."
)


def risk_manager_node(state: CommitteeState) -> dict:
    quant = state["quant"]
    sent = state["sentiment"]

    exp_return = quant["forecast_7d_pct"] / 100.0
    # 0 is its own case: a flat forecast has no direction, and folding it into
    # the short side would report a SELL-shaped disagreement for a non-signal.
    quant_dir = 1 if exp_return > 0 else -1 if exp_return < 0 else 0
    sent_score = sent["sentiment_now"]
    sent_dir = (1 if sent_score >= config.SENTIMENT_STRONG
                else -1 if sent_score <= -config.SENTIMENT_STRONG else 0)

    if quant_dir == 0:
        # No directional forecast to agree or disagree with. The materiality
        # gate below turns this into a HOLD regardless.
        agreement = "no_signal"
    elif sent_dir == 0:
        agreement = "sentiment_neutral"
    elif sent_dir == quant_dir:
        agreement = "aligned"
    else:
        agreement = "conflicting"

    # ---- deterministic directive ----------------------------------------
    # Rule 1 (signal confirmation): two independent streams must agree.
    # Rule 2 (drawdown discipline): the move must exceed horizon noise.
    # Rule 3 (regime awareness): high volatility halves conviction.
    # Two gates only, and they are not redundant with each other:
    #   Gate 1 (conviction)   -- the forecast must be strong relative to the
    #                            volatility of its own horizon. Signal-to-noise
    #                            already encodes magnitude, so a separate
    #                            absolute return threshold would gate the same
    #                            quantity twice and simply mean never trading.
    #   Gate 2 (confirmation) -- the two independent evidence streams must not
    #                            contradict each other.
    # Direction then comes from the sign of the forecast.
    reasons: list[str] = []
    tradable = quant["signal_to_noise"] >= config.SIGNAL_TO_NOISE_MIN
    material = abs(exp_return) >= config.MIN_MATERIAL_RETURN

    if not material:
        directive = "HOLD"
        reasons.append(
            f"the projected {exp_return:+.2%} move is numerically negligible")
    elif not tradable:
        directive = "HOLD"
        reasons.append(
            f"the forecast's signal-to-noise ratio of {quant['signal_to_noise']:.2f} "
            f"falls below the {config.SIGNAL_TO_NOISE_MIN:.2f} conviction floor "
            "calibrated on validation data, so this is not among the days the "
            "committee is willing to act on")
    elif agreement == "conflicting":
        directive = "HOLD"
        reasons.append(
            "the quantitative forecast and the news sentiment point in opposite "
            "directions; under the signal-confirmation rule, disagreement between "
            "independent evidence streams is itself information about regime "
            "instability and calls for no position")
    else:
        directive = "BUY" if quant_dir > 0 else "SELL"
        reasons.append(
            f"the {exp_return:+.2%} forecast clears the conviction floor with a "
            f"signal-to-noise ratio of {quant['signal_to_noise']:.2f}, placing it "
            "in the top quintile of the validation distribution")

    if agreement == "aligned" and directive != "HOLD":
        reasons.append("the sentiment engine independently confirms the direction")

    # Conviction, then the risk overlays that can only reduce it.
    conviction = 0.5
    if directive != "HOLD":
        # Scaled against the conviction floor: a signal at twice the floor
        # earns near-full conviction, one just above it earns little.
        ratio = quant["signal_to_noise"] / config.SIGNAL_TO_NOISE_MIN
        conviction = min(0.95, 0.35 + 0.30 * min(ratio, 2.0))
        if agreement == "aligned":
            conviction = min(0.95, conviction + 0.15)

    position_pct = 0.0 if directive == "HOLD" else round(conviction * 60, 1)

    if quant["volatility_regime"] == "high" and position_pct > 0:
        position_pct = round(position_pct / 2, 1)
        reasons.append(
            "position size is halved because realised volatility is in the top "
            "quintile of its trailing distribution, per the regime-awareness rule")

    trap = None
    if agreement == "conflicting":
        if quant_dir > 0 and sent_dir < 0:
            trap = ("**Possible bear trap.** Sentiment is markedly negative while "
                    "the quantitative trajectory remains positive — the classic "
                    "signature of panic selling into an intact trend.")
        else:
            trap = ("**Possible bull trap.** Sentiment is markedly positive while "
                    "the quantitative trajectory points down — the classic "
                    "signature of a relief rally inside a weakening structure.")
    if sent["contrarian_flag"] and directive != "HOLD":
        reasons.append("one-sided news coverage argues for restraint on size")

    observations = _risk_observations(directive, agreement, quant, sent, position_pct)
    principles = retrieve_governing_principles(agreement, quant, sent)

    findings = {
        "as_of": quant["as_of"],
        "directive": directive,
        "conviction": round(conviction, 3),
        "position_pct": position_pct,
        "agreement": agreement,
        "tradable": tradable,
        "trap_warning": trap,
        "reasons": reasons,
        "observations": observations,
        "principles": principles,
        "invalidation": {
            "horizon_days": config.LSTM.horizon,
            "expected_move_pct": quant["forecast_7d_pct"],
            "noise_band_pct": quant["horizon_volatility_pct"],
        },
    }

    principle_block = "\n".join(
        f"- **{p['principle']}** ({p['source']}): {p['text']}"
        for p in principles) or "- (no principle matched this decision state)"
    fallback = _render_risk_report(findings, quant, sent)
    prose = _narrate(
        state,
        "risk_manager",
        RISK_MANAGER_SYSTEM,
        _brief(
            ("What to write", RISK_TASK),
            ("The computed directive",
             f"- directive: {directive}\n"
             f"- position size: {position_pct}% of standard\n"
             f"- conviction: {conviction:.0%}\n"
             f"- agreement between the two streams: {agreement.replace('_', ' ')}"),
            ("Why the rules produced it",
             "\n".join(f"- {r}" for r in reasons)),
            ("How close it came to another call", _fmt_observations(observations)),
            ("Governing principles retrieved for this state", principle_block),
            ("Invalidation frame", _fmt_fields(findings["invalidation"])),
            ("Upstream: quantitative findings",
             _fmt_fields(quant, skip=("forecast_prices",))),
            ("Upstream: quantitative cross-checks",
             _fmt_observations(quant.get("observations"))),
            ("Upstream: sentiment findings",
             _fmt_fields(sent, skip=("retrieval",))),
            ("Upstream: sentiment cross-checks",
             _fmt_observations(sent.get("observations"))),
            # The headlines themselves, not just the scores computed from them.
            # A directive justified against "sentiment_now: -0.16" is justified
            # against a number; justified against the coverage that produced it,
            # it can say what the market was actually arguing about.
            ("Upstream: the retrieved headlines", _fmt_evidence(sent.get("evidence"))),
            ("Upstream: how the news evidence was retrieved",
             _fmt_fields(sent.get("retrieval", {}))),
        ),
        fallback,
    )
    prose = guard_directive(prose, fallback, directive)
    return {"risk": findings, "reports": {"risk_manager": prose}}


# A model asserting the call, rather than mentioning one. The committee's own
# cross-checks legitimately name the *other* directive -- "had the forecast
# cleared the floor, the directive would have been BUY" -- so a bare search for
# the word finds the counterfactual and flags a report that is doing its job.
# Only a claim about what the call *is* counts.
_DIRECTIVE_CLAIM = re.compile(
    r"\b(?:directive|recommendation|verdict|call|rating|stance|position)\b"
    r"[^.\n]{0,40}?"
    r"\b(?:is|are|remains?|stands? at|:)\s*"
    r"[*`_\s]*\b(BUY|SELL|HOLD)\b",
    re.IGNORECASE)


def directive_contradiction(prose: str, directive: str) -> str | None:
    """Return why ``prose`` disagrees with ``directive``, or ``None`` if it does not.

    Narration is the one part of the report a language model writes, and the
    directive is the one thing it must not restate wrongly. Two failures are
    caught: asserting a different call, and never naming the real one -- a
    justification that cannot bring itself to say the word is not a
    justification of it.
    """
    claimed = {m.group(1).upper() for m in _DIRECTIVE_CLAIM.finditer(prose)}
    wrong = claimed - {directive.upper()}
    if wrong:
        return f"prose asserts {'/'.join(sorted(wrong))}"
    if not re.search(rf"\b{re.escape(directive)}\b", prose, re.IGNORECASE):
        return f"prose never names the {directive} directive"
    return None


def guard_directive(prose: str, fallback: str, directive: str) -> str:
    """Return ``prose``, or the deterministic report if the prose disagrees.

    The prose is the only part of this report a model wrote, and it is the part
    a reader trusts most. A narration that names a different call than the rules
    did is worse than no narration at all, so the template stands in -- the same
    failure posture the narrative layer takes for an API outage.
    """
    reason = directive_contradiction(prose, directive)
    if reason is None:
        return prose
    log.warning("Risk Manager narration contradicts the %s directive (%s); "
                "falling back to the deterministic report", directive, reason)
    return fallback


def _risk_observations(directive: str, agreement: str, quant: dict, sent: dict,
                       position_pct: float) -> list[dict]:
    """What the directive rests on, and how close it came to being another one.

    "HOLD" on its own tells the reader nothing about whether the call was
    marginal or nowhere near. These quantify the gap, and carry any integrity
    flag the upstream agents raised down into the final directive -- the Risk
    Manager sees both analyses, so it is the right place to say that a directive
    rests on numbers the Data Analyst already doubted.
    """
    out: list[dict] = []
    snr, floor = quant["signal_to_noise"], config.SIGNAL_TO_NOISE_MIN
    exp_return = quant["forecast_7d_pct"] / 100.0

    # An integrity problem upstream outranks everything else here.
    upstream = [o for src in (quant, sent) for o in src.get("observations", [])
                if o["severity"] == "integrity"]
    if upstream:
        out.append({
            "kind": "suspect_inputs", "severity": "integrity",
            "text": (f"{'this directive rests on' if directive != 'HOLD' else 'the abstention is safe, but'} "
                     f"{len(upstream)} input{'s' if len(upstream) > 1 else ''} the "
                     "committee has already flagged as suspect. Reconcile the data "
                     "before acting on any call from this session"),
        })

    # How marginal was it?
    if directive == "HOLD" and snr < floor:
        if snr > 1e-9:
            needed = abs(exp_return) * floor / snr
            out.append({
                "kind": "distance_to_gate", "severity": "note",
                "text": (f"this was not a near miss: the forecast would have to be "
                         f"{floor / snr:.1f}x larger — about {needed:.2%} over the "
                         f"week instead of {abs(exp_return):.2%} — to clear the "
                         f"{floor:.2f} conviction floor"
                         if floor / snr >= 1.5 else
                         f"this was close: a forecast {floor / snr:.2f}x larger, "
                         f"about {needed:.2%} instead of {abs(exp_return):.2%}, "
                         "would have cleared the conviction floor"),
            })
        # What the committee would have said had the gate passed.
        if agreement != "conflicting" and exp_return != 0:
            would = "BUY" if exp_return > 0 else "SELL"
            out.append({
                "kind": "counterfactual", "severity": "note",
                "text": (f"nothing else was blocking it — had the forecast cleared "
                         f"the floor, the directive would have been {would}"),
            })
    elif directive == "HOLD" and agreement == "conflicting":
        out.append({
            "kind": "counterfactual", "severity": "note",
            "text": (f"the conviction gate was cleared at {snr:.2f} against a "
                     f"{floor:.2f} floor, so this is a HOLD on disagreement alone: "
                     "the forecast was strong enough to act on and the sentiment "
                     "channel refused to confirm it"),
        })
    elif directive != "HOLD":
        out.append({
            "kind": "sizing", "severity": "note",
            "text": (f"the position is {position_pct:.1f}% of standard rather than "
                     "full size, because conviction scales with how far the signal "
                     f"clears the floor and this one clears it {snr / floor:.2f}x"),
        })
    return out


def principle_queries(agreement: str, quant: dict, sent: dict) -> list[str]:
    """One short query per concern the directive actually rests on.

    A single combined query does not work here. The old one appended a constant
    "position sizing and confirmation requirements" to every request, and that
    tail dominated the embedding: across all 36 reachable decision states the
    same two principles came back every time, only 8 distinct top-3 sets were
    ever produced, and 8 of the 15 principles were never retrieved at all --
    including Bull Trap and Bear Trap, which this very node diagnoses two dozen
    lines above.

    Asking one focused question per concern instead keeps each concern
    represented rather than letting the loudest phrase win. Note this affects
    only the principles cited in the write-up: the directive is already decided
    by the time we get here.
    """
    queries = []

    # 1. Sizing. Always asked -- position_pct rests on it either way.
    if quant["volatility_regime"] == "high":
        queries.append("realised volatility is in the top quintile of its "
                       "trailing distribution; cut position size and widen "
                       "forecast intervals")
    else:
        queries.append("how large a position does a forecast with a narrow "
                       "expected edge justify")

    # 2. What the two evidence streams are doing.
    queries.append({
        "conflicting": "the price forecast and the news sentiment point in "
                       "opposite directions; bull trap or bear trap",
        "aligned": "two independent evidence streams confirm the same direction",
        "no_signal": "the forecast is flat and provides no direction to act on",
    }.get(agreement,
          "news sentiment is neutral and offers no confirmation of the forecast"))

    # 3. Crowding, only when the sentiment critic actually raised it.
    if sent.get("contrarian_flag"):
        queries.append("news coverage is uniformly one-sided; crowded attention "
                       "and contrarian warning")
    return queries


def retrieve_governing_principles(agreement: str, quant: dict, sent: dict,
                                  per_query: int = 2, cap: int = 4) -> list[dict]:
    """Principles for the write-up: top ``per_query`` per concern, de-duplicated."""
    retriever = get_retriever()
    seen: set[str] = set()
    out: list[dict] = []
    for query in principle_queries(agreement, quant, sent):
        for principle in retriever.retrieve_principles(query, top_k=per_query):
            name = principle.get("principle")
            if name not in seen:
                seen.add(name)
                out.append(principle)
    return out[:cap]


def _render_risk_report(f: dict, quant: dict, sent: dict) -> str:
    lines = [
        "### Risk Manager — Final Directive",
        "",
        f"## `{f['directive']}`  ·  position size **{f['position_pct']:.1f}%** of standard",
        "",
        f"The quantitative and sentiment streams are **{f['agreement'].replace('_', ' ')}**. "
        f"This directive follows because " + "; ".join(f["reasons"]) + ".",
        "",
        f"**Invalidation frame.** The call is scoped to {f['invalidation']['horizon_days']} "
        f"trading sessions with an expected move of "
        f"{f['invalidation']['expected_move_pct']:+.2f}% against a noise band of "
        f"{f['invalidation']['noise_band_pct']:.2f}%. If realised volatility exceeds that "
        "band, the thesis is void and the position should be closed regardless of direction.",
    ]
    if f["trap_warning"]:
        lines += ["", f"> {f['trap_warning']}"]

    obs = f.get("observations", [])
    integrity = [o for o in obs if o["severity"] == "integrity"]
    notes = [o for o in obs if o["severity"] != "integrity"]
    if integrity:
        lines += ["", "> **Inputs are suspect.** "
                  + " Also, ".join(o["text"] for o in integrity) + "."]
    if notes:
        lines += ["", "**How close this was to another call:** "
                  + "; ".join(o["text"] for o in notes) + "."]
    lines += ["", "**Governing principles consulted:**"]
    for p in f["principles"]:
        # Cited in full. Every chunk in the knowledge base is one short
        # paragraph (148-348 characters), so the old 200-character cut severed
        # most of them mid-sentence -- and a principle that stops before its
        # own conclusion cannot justify the directive it is cited for.
        lines.append(f"- **{p['principle']}** — {p['text']}"
                     if p["text"] else f"- **{p['principle']}**")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
_FORECASTER: ForecastService | None = None


def _forecast_service() -> ForecastService:
    global _FORECASTER
    if _FORECASTER is None:
        _FORECASTER = ForecastService()
    return _FORECASTER


def build_graph():
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(CommitteeState)
    g.add_node("data_analyst", data_analyst_node)
    g.add_node("sentiment_critic", sentiment_critic_node)
    g.add_node("risk_manager", risk_manager_node)

    # Sequential: the risk manager must see both upstream analyses to detect
    # the disagreement that drives the trap diagnosis.
    g.add_edge(START, "data_analyst")
    g.add_edge("data_analyst", "sentiment_critic")
    g.add_edge("sentiment_critic", "risk_manager")
    g.add_edge("risk_manager", END)
    return g.compile()


_GRAPH = None


def run_committee(features: pd.DataFrame, as_of: str | pd.Timestamp,
                  narrative: bool = True, brief_dir: str | None = None,
                  narration_dir: str | None = None) -> dict:
    """Run the full committee for one decision date and return its state.

    ``narrative=False`` skips the language model entirely and renders the
    deterministic templates instead -- what the backtest wants, since it reads
    only the findings. Directives are identical either way; the model never
    touches them.

    ``brief_dir`` writes each agent's narration brief out; ``narration_dir``
    reads the prose back from files rather than calling a model. Together they
    let the narration be produced anywhere -- another provider, or an assistant
    session -- while the findings and the directive stay in the pipeline.
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    as_of = pd.Timestamp(as_of)
    return _GRAPH.invoke({"as_of": str(as_of.date()), "features": features,
                          "narrative": narrative, "brief_dir": brief_dir,
                          "narration_dir": narration_dir})


def executive_report(state: dict) -> str:
    """Assemble the committee's markdown executive investment report."""
    risk, quant, sent = state["risk"], state["quant"], state["sentiment"]
    header = [
        "# FinAgent-Pulse — Executive Investment Report",
        f"**Asset:** {config.TICKER_LABEL} ({config.TICKER})  ·  "
        f"**Decision date:** {risk['as_of']}  ·  "
        f"**Horizon:** {config.LSTM.horizon} trading sessions",
        "",
        "| Directive | Position | 7-day forecast | Sentiment | Regime |",
        "|---|---|---|---|---|",
        f"| **{risk['directive']}** | {risk['position_pct']:.1f}% | "
        f"{quant['forecast_7d_pct']:+.2f}% | "
        f"{sent['sentiment_now']:+.3f} ({sent['stance']}) | "
        f"{quant['volatility_regime']} vol |",
        "",
        "---",
        "",
    ]
    body = [state["reports"][k] for k in
            ("data_analyst", "sentiment_critic", "risk_manager")
            if k in state["reports"]]
    footer = [
        "",
        "---",
        "",
        f"*Generated by FinAgent-Pulse. Narrative mode: "
        f"`{narration_mode(state)}`. Directives are computed deterministically "
        "from model output and are reproducible. This is an academic prototype, "
        "not investment advice.*",
    ]
    return "\n".join(header + ["\n\n".join(body)] + footer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from finagent_pulse.data.preprocess import merge_features

    feats = merge_features()
    st = run_committee(feats, feats["date"].iloc[-1])
    print(executive_report(st))
