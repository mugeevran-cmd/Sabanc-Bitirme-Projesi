"""The observations the agents narrate on top of their raw findings.

The templates used to relay each number and stop, which left the reader to spot
for themselves that a forecast was larger than anything the index has ever done,
or that a Fear & Greed reading in the greed band sat next to a session whose own
headlines were net negative. Those cross-checks are now computed, which means
they can be pinned -- and they have to be, because a template that quietly stops
firing still renders a perfectly plausible-looking report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finagent_pulse import config
from finagent_pulse.agents.committee import (
    _fmt_evidence, _fmt_fields, _fmt_observations, _quant_observations, _render_quant_report,
    _render_risk_report, _render_sentiment_report, _risk_observations,
    _sentiment_observations, directive_contradiction, guard_directive)

FLOOR = config.SIGNAL_TO_NOISE_MIN


def kinds(obs: list[dict]) -> set[str]:
    return {o["kind"] for o in obs}


@pytest.fixture
def hist() -> pd.DataFrame:
    """300 sessions of a ~1%/day random walk — ordinary index-sized moves."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"close": 4000 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))})


def quant(**over) -> dict:
    f = {"forecast_7d_pct": 0.5, "forecast_path_pct": [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5],
         "rsi_14": 50.0, "dist_ma50_pct": 1.0,
         "signal_to_noise": 0.21, "volatility_regime": "normal", "as_of": "2023-06-01"}
    f.update(over)
    return f


# --------------------------------------------------------------------------
# Data Analyst — the check that catches a mis-scaled forecast
# --------------------------------------------------------------------------
def test_a_forecast_outside_the_realised_distribution_is_an_integrity_flag(hist):
    """The failure this exists for: a −22% weekly index forecast is not a forecast."""
    obs = _quant_observations(quant(forecast_7d_pct=-22.0,
                                    forecast_path_pct=[-3, -6, -9, -12, -16, -19, -22]), hist)
    scale = [o for o in obs if o["kind"] == "forecast_scale"]
    assert scale and scale[0]["severity"] == "integrity"
    assert "data or scaling fault" in scale[0]["text"]


def test_an_ordinary_forecast_raises_nothing(hist):
    assert "forecast_scale" not in kinds(_quant_observations(quant(), hist))


def test_a_large_but_plausible_forecast_is_a_note_not_an_integrity_flag(hist):
    """5% sits at the 92nd percentile of this walk — big, but the index does it.

    There are three bands, and only the top one is an integrity problem: a move
    the index makes a few times a year should read as "large", not as "broken".
    """
    obs = _quant_observations(quant(forecast_7d_pct=5.0), hist)
    scale = [o for o in obs if o["kind"] == "forecast_scale"]
    assert scale and scale[0]["severity"] == "note"


def test_too_little_history_skips_the_scale_check():
    """No trailing distribution to compare against -- say nothing rather than guess."""
    short = pd.DataFrame({"close": np.linspace(4000, 4100, 30)})
    assert "forecast_scale" not in kinds(_quant_observations(quant(forecast_7d_pct=-22.0), short))


def test_a_round_trip_inside_the_horizon_is_reported(hist):
    """+2% by midweek and +0.1% by Friday is not a "+0.1% week"."""
    obs = _quant_observations(
        quant(forecast_7d_pct=0.1,
              forecast_path_pct=[0.5, 1.2, 2.0, 1.5, 0.8, 0.3, 0.1]), hist)
    assert "path_shape" in kinds(obs)
    assert "not monotone" in next(o["text"] for o in obs if o["kind"] == "path_shape")


def test_a_monotone_path_is_not_flagged(hist):
    assert "path_shape" not in kinds(_quant_observations(quant(), hist))


@pytest.mark.parametrize("rsi, forecast", [(75.0, 1.0), (25.0, -1.0)])
def test_the_model_asking_to_chase_an_extreme_is_called_out(hist, rsi, forecast):
    obs = _quant_observations(quant(rsi_14=rsi, forecast_7d_pct=forecast), hist)
    assert "momentum_conflict" in kinds(obs)


def test_price_far_from_its_own_moving_average_is_an_integrity_flag(hist):
    """The other half of the screenshot bug: price 27% above its MAs."""
    obs = _quant_observations(quant(dist_ma50_pct=27.0), hist)
    ma = [o for o in obs if o["kind"] == "ma_dislocation"]
    assert ma and ma[0]["severity"] == "integrity"


# --------------------------------------------------------------------------
# Sentiment Critic
# --------------------------------------------------------------------------
def sent(**over) -> dict:
    f = {"sentiment_now": 0.10, "fear_greed_index": 50.0, "sentiment_shift": 0.0,
         "headline_count": 30, "evidence": [{"sentiment": 0.1}]}
    f.update(over)
    return f


def test_greed_reading_beside_negative_headlines_is_explained_not_hidden():
    """The real 2024-03-04 case: F&G 88/100 while the session scores −0.161."""
    obs = _sentiment_observations(sent(fear_greed_index=88.0, sentiment_now=-0.161))
    turn = [o for o in obs if o["kind"] == "regime_turn"]
    assert turn
    assert "20-day smoothed percentile" in turn[0]["text"]


def test_fear_reading_beside_positive_headlines_is_the_mirror_case():
    assert "regime_turn" in kinds(
        _sentiment_observations(sent(fear_greed_index=20.0, sentiment_now=0.20)))


def test_an_agreeing_channel_says_so():
    assert _sentiment_observations(sent(fear_greed_index=70.0, sentiment_now=0.20)) == []


def test_a_handful_of_headlines_is_not_a_measurement():
    obs = _sentiment_observations(sent(headline_count=3))
    thin = [o for o in obs if o["kind"] == "thin_coverage"]
    assert thin and thin[0]["severity"] == "integrity"


def test_retrieved_evidence_disagreeing_in_sign_is_reported():
    obs = _sentiment_observations(
        sent(sentiment_now=-0.30, evidence=[{"sentiment": 0.4}, {"sentiment": 0.6}]))
    assert "evidence_divergence" in kinds(obs)


def test_a_near_zero_session_does_not_trigger_evidence_divergence():
    """Sign of a number that is essentially zero is not information."""
    assert "evidence_divergence" not in kinds(
        _sentiment_observations(sent(sentiment_now=0.01, evidence=[{"sentiment": -0.4}])))


def test_a_moving_baseline_is_reported_as_direction_of_travel():
    obs = _sentiment_observations(sent(sentiment_shift=-0.25))
    assert "deteriorating" in next(o["text"] for o in obs if o["kind"] == "momentum")


# --------------------------------------------------------------------------
# Risk Manager
# --------------------------------------------------------------------------
def test_an_upstream_integrity_flag_reaches_the_directive():
    """The Risk Manager sees both analyses, so it is where this has to surface."""
    q = quant(observations=[{"kind": "forecast_scale", "severity": "integrity",
                             "text": "the projected move is out of scale"}])
    obs = _risk_observations("HOLD", "sentiment_neutral", q, sent(), 0.0)
    flagged = [o for o in obs if o["kind"] == "suspect_inputs"]
    assert flagged and flagged[0]["severity"] == "integrity"


def test_clean_inputs_raise_no_suspicion():
    assert "suspect_inputs" not in kinds(
        _risk_observations("HOLD", "sentiment_neutral", quant(), sent(), 0.0))


def test_a_marginal_hold_says_how_marginal():
    q = quant(signal_to_noise=FLOOR * 0.9, forecast_7d_pct=0.5)
    text = next(o["text"] for o in _risk_observations("HOLD", "sentiment_neutral", q, sent(), 0.0)
                if o["kind"] == "distance_to_gate")
    assert "this was close" in text


def test_a_hold_that_was_nowhere_near_says_that_instead():
    q = quant(signal_to_noise=FLOOR * 0.2, forecast_7d_pct=0.1)
    text = next(o["text"] for o in _risk_observations("HOLD", "sentiment_neutral", q, sent(), 0.0)
                if o["kind"] == "distance_to_gate")
    assert "not a near miss" in text


def test_the_counterfactual_directive_is_named():
    q = quant(signal_to_noise=FLOOR * 0.9, forecast_7d_pct=0.5)
    obs = _risk_observations("HOLD", "sentiment_neutral", q, sent(), 0.0)
    assert "would have been BUY" in next(
        o["text"] for o in obs if o["kind"] == "counterfactual")


def test_a_hold_on_disagreement_is_distinguished_from_a_hold_on_conviction():
    """Two very different HOLDs that used to render identically."""
    q = quant(signal_to_noise=FLOOR * 1.5)
    text = next(o["text"] for o in _risk_observations("HOLD", "conflicting", q, sent(), 0.0)
                if o["kind"] == "counterfactual")
    assert "disagreement alone" in text


def test_a_traded_directive_explains_its_size():
    obs = _risk_observations("BUY", "aligned", quant(), sent(), 45.0)
    assert "sizing" in kinds(obs)


# --------------------------------------------------------------------------
# The renderers actually print them
# --------------------------------------------------------------------------
def full_quant(obs) -> dict:
    return dict(quant(), close=4000.0, trend_direction="up", volatility_percentile=0.4,
                horizon_volatility_pct=1.9, anomalies=[], observations=obs)


def test_integrity_flags_are_rendered_as_a_warning():
    r = _render_quant_report(full_quant(
        [{"kind": "forecast_scale", "severity": "integrity", "text": "the move is out of scale"}]))
    assert "Data integrity" in r and "the move is out of scale" in r


def test_notes_are_rendered_separately_from_integrity_flags():
    r = _render_quant_report(full_quant(
        [{"kind": "path_shape", "severity": "note", "text": "the path round-trips"}]))
    assert "Reading the forecast" in r and "Data integrity" not in r


def test_a_clean_forecast_gets_an_explicit_all_clear():
    assert "internally consistent" in _render_quant_report(full_quant([]))


def test_the_risk_report_prints_its_observations():
    f = {"directive": "HOLD", "position_pct": 0.0, "conviction": 0.5,
         "agreement": "sentiment_neutral", "reasons": ["a reason"], "trap_warning": None,
         "principles": [], "invalidation": {"horizon_days": 7, "expected_move_pct": 0.26,
                                            "noise_band_pct": 1.9},
         "observations": [{"kind": "counterfactual", "severity": "note",
                           "text": "it would have been BUY"}]}
    assert "it would have been BUY" in _render_risk_report(f, quant(), sent())


def test_the_sentiment_report_prints_its_observations():
    f = dict(sent(), as_of="2024-03-04", stance="neutral", sentiment_5d=0.0,
             sentiment_20d_baseline=0.0, coverage_zscore=0.0, crowded_attention=False,
             contrarian_flag=None, top_drivers=[], retrieval={},
             observations=[{"kind": "regime_turn", "severity": "note",
                            "text": "the index and the session disagree"}])
    assert "the index and the session disagree" in _render_sentiment_report(f, [])


# --------------------------------------------------------------------------
# The brief the narrator is given
# --------------------------------------------------------------------------
# The prose is only as good as what the model is handed. These pin the two ways
# the brief can quietly stop carrying the committee's reasoning: dropping the
# argument, or burying an integrity flag below the routine notes.
def test_the_brief_leaves_the_narrative_fields_to_their_own_sections():
    """`reasons` and `observations` are the argument -- not `key: value` filler."""
    fields = _fmt_fields({"directive": "HOLD", "conviction": 0.5,
                          "reasons": ["a reason"], "observations": [{"text": "x"}],
                          "principles": [{"principle": "p"}], "evidence": [{"headline": "h"}]})
    assert "directive: HOLD" in fields
    assert "a reason" not in fields and "principle" not in fields and "headline" not in fields


def test_the_brief_drops_the_caller_s_skipped_fields():
    assert "close" not in _fmt_fields({"close": 4000.0, "rsi_14": 50.0}, skip=("close",))


def test_integrity_flags_lead_the_cross_check_section():
    """A note printed above an integrity flag buries the thing that matters."""
    rendered = _fmt_observations([
        {"kind": "path_shape", "severity": "note", "text": "the path round-trips"},
        {"kind": "forecast_scale", "severity": "integrity", "text": "the move is out of scale"}])
    assert rendered.index("out of scale") < rendered.index("round-trips")


def test_no_cross_checks_says_so_rather_than_going_blank():
    """An empty section reads as missing data; the all-clear is a finding."""
    assert "cross-check cleanly" in _fmt_observations([])


# --------------------------------------------------------------------------
# The narration cannot overrule the rules
# --------------------------------------------------------------------------
def test_prose_claiming_a_different_call_is_caught():
    assert directive_contradiction("Recommendation: BUY at full size.", "HOLD")


@pytest.mark.parametrize("prose", [
    "The directive is HOLD.",
    "Our **call: HOLD** this week.",
    "The committee's verdict remains HOLD until the gate clears.",
])
def test_prose_agreeing_with_the_directive_passes(prose):
    assert directive_contradiction(prose, "HOLD") is None


def test_the_committee_s_own_counterfactual_is_not_a_contradiction():
    """The real false positive: `_risk_observations` names the other directive.

    "had the forecast cleared the floor, the directive would have been BUY" is
    the report doing its job. A bare search for the word BUY flags it, which
    would make the guard fire on exactly the reports worth reading.
    """
    prose = ("The call is HOLD. Nothing else was blocking it -- had the forecast "
             "cleared the floor, the directive would have been BUY.")
    assert directive_contradiction(prose, "HOLD") is None


def test_prose_that_never_names_the_directive_is_caught():
    """A justification that cannot say the word is not a justification of it."""
    assert directive_contradiction("Markets are calm, so we wait.", "HOLD")


def test_a_contradicting_narration_is_replaced_by_the_deterministic_report():
    assert guard_directive("We recommend BUY.", "TEMPLATE", "HOLD") == "TEMPLATE"


def test_a_sound_narration_is_kept():
    assert guard_directive("The directive is HOLD.", "TEMPLATE", "HOLD") == "The directive is HOLD."


def test_the_brief_carries_the_headlines_with_the_arms_that_found_them():
    """The Risk Manager justifies against coverage, not just against a mean score."""
    rendered = _fmt_evidence([{"date": "2024-03-01", "label": "negative",
                               "sentiment": -0.3747, "provenance": "bm25+vector",
                               "headline": "Is S&P 500 in a bubble zone?"}])
    assert "bubble zone" in rendered and "bm25+vector" in rendered and "-0.37" in rendered


def test_an_empty_retrieval_is_named_rather_than_left_blank():
    assert "nothing retrieved" in _fmt_evidence([])
