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
    quant: Annotated[dict, _merge]
    sentiment: Annotated[dict, _merge]
    risk: Annotated[dict, _merge]
    reports: Annotated[dict, _merge]


# --------------------------------------------------------------------------
# Agent 1 -- Data Analyst
# --------------------------------------------------------------------------
DATA_ANALYST_SYSTEM = (
    "You are a quantitative data analyst on an institutional investment "
    "committee. You interpret model output and price structure. You never "
    "invent numbers: every figure you cite must appear in the findings given "
    "to you. Write 3-5 tight paragraphs of markdown, no preamble."
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

    fallback = _render_quant_report(findings)
    prose = get_writer().write(
        DATA_ANALYST_SYSTEM,
        "Explain these quantitative findings for the investment committee. "
        "Be explicit about whether the forecast is large enough to trade "
        f"against 7-day noise.\n\nFINDINGS:\n{findings}",
        fallback,
    )
    return {"quant": findings, "reports": {"data_analyst": prose}}


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

    evidence_block = "\n".join(
        f"- [{d.date}] ({d.label}, {d.sentiment:+.2f}) {d.headline}" for d in docs)
    fallback = _render_sentiment_report(findings, docs)
    prose = get_writer().write(
        SENTIMENT_CRITIC_SYSTEM,
        "Assess the news environment for the committee. Identify the core "
        "drivers and say whether this looks like genuine information or crowd "
        f"reaction.\n\nFINDINGS:\n{findings}\n\nRETRIEVED EVIDENCE:\n{evidence_block}",
        fallback,
    )
    return {"sentiment": findings, "reports": {"sentiment_critic": prose}}


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
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Agent 3 -- Risk Manager
# --------------------------------------------------------------------------
RISK_MANAGER_SYSTEM = (
    "You are the risk manager on an institutional investment committee and you "
    "hold the final word. You are deliberately conservative: capital "
    "preservation outranks opportunity. You have been given a directive that "
    "was computed by rule; explain and justify it against the cited investment "
    "principles. Do not contradict the directive. Write 3-5 tight paragraphs "
    "of markdown, no preamble."
)


def risk_manager_node(state: CommitteeState) -> dict:
    quant = state["quant"]
    sent = state["sentiment"]

    exp_return = quant["forecast_7d_pct"] / 100.0
    quant_dir = 1 if exp_return > 0 else -1
    sent_score = sent["sentiment_now"]
    sent_dir = (1 if sent_score >= config.SENTIMENT_STRONG
                else -1 if sent_score <= -config.SENTIMENT_STRONG else 0)

    agreement = "aligned" if sent_dir != 0 and sent_dir == quant_dir else \
                "conflicting" if sent_dir != 0 and sent_dir != quant_dir else \
                "sentiment_neutral"

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

    principle_query = (
        f"{directive} decision with {agreement} signals in a "
        f"{quant['volatility_regime']} volatility regime; position sizing and "
        "confirmation requirements")
    principles = get_retriever().retrieve_principles(principle_query, top_k=3)

    findings = {
        "as_of": quant["as_of"],
        "directive": directive,
        "conviction": round(conviction, 3),
        "position_pct": position_pct,
        "agreement": agreement,
        "tradable": tradable,
        "trap_warning": trap,
        "reasons": reasons,
        "principles": principles,
        "invalidation": {
            "horizon_days": config.LSTM.horizon,
            "expected_move_pct": quant["forecast_7d_pct"],
            "noise_band_pct": quant["horizon_volatility_pct"],
        },
    }

    principle_block = "\n".join(
        f"- **{p['principle']}** ({p['source']}): {p['text'][:260]}" for p in principles)
    fallback = _render_risk_report(findings, quant, sent)
    prose = get_writer().write(
        RISK_MANAGER_SYSTEM,
        f"The committee's computed directive is {directive} at "
        f"{position_pct}% of standard position size. Justify it.\n\n"
        f"QUANT:\n{ {k: v for k, v in quant.items() if k != 'forecast_prices'} }\n\n"
        f"SENTIMENT:\n{ {k: v for k, v in sent.items() if k not in ('evidence', 'retrieval')} }\n\n"
        f"RISK FINDINGS:\n{ {k: v for k, v in findings.items() if k != 'principles'} }\n\n"
        f"APPLICABLE PRINCIPLES:\n{principle_block}",
        fallback,
    )
    return {"risk": findings, "reports": {"risk_manager": prose}}


def _render_risk_report(f: dict, quant: dict, sent: dict) -> str:
    lines = [
        "### Risk Manager — Final Directive",
        "",
        f"## `{f['directive']}`  ·  position size **{f['position_pct']:.1f}%** of standard  ·  "
        f"conviction **{f['conviction']:.0%}**",
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
    lines += ["", "**Governing principles consulted:**"]
    for p in f["principles"]:
        lines.append(f"- **{p['principle']}** — {p['text'][:200].rstrip()}...")
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


def run_committee(features: pd.DataFrame, as_of: str | pd.Timestamp) -> dict:
    """Run the full committee for one decision date and return its state."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    as_of = pd.Timestamp(as_of)
    return _GRAPH.invoke({"as_of": str(as_of.date()), "features": features})


def executive_report(state: dict) -> str:
    """Assemble the committee's markdown executive investment report."""
    risk, quant, sent = state["risk"], state["quant"], state["sentiment"]
    header = [
        f"# FinAgent-Pulse — Executive Investment Report",
        f"**Asset:** {config.TICKER_LABEL} ({config.TICKER})  ·  "
        f"**Decision date:** {risk['as_of']}  ·  "
        f"**Horizon:** {config.LSTM.horizon} trading sessions",
        "",
        f"| Directive | Position | Conviction | 7-day forecast | Sentiment | Regime |",
        f"|---|---|---|---|---|---|",
        f"| **{risk['directive']}** | {risk['position_pct']:.1f}% | "
        f"{risk['conviction']:.0%} | {quant['forecast_7d_pct']:+.2f}% | "
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
        f"`{get_writer().mode}`. Directives are computed deterministically from "
        "model output and are reproducible. This is an academic prototype, not "
        "investment advice.*",
    ]
    return "\n".join(header + ["\n\n".join(body)] + footer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from finagent_pulse.data.preprocess import merge_features

    feats = merge_features()
    st = run_committee(feats, feats["date"].iloc[-1])
    print(executive_report(st))
