"""The Risk Manager's decision rule.

This is the part of the system that actually issues BUY / SELL / HOLD, and the
project's central claim about it is that it is deterministic and auditable --
computed in Python, never by the language model. That claim is only worth
making if the rule is pinned down, which is what these tests do.
"""
from __future__ import annotations

import pytest

from finagent_pulse import config
from finagent_pulse.agents.committee import risk_manager_node


def decide(quant: dict, sent: dict) -> dict:
    """Run the rule with narration off and return the risk findings."""
    return risk_manager_node({"quant": quant, "sentiment": sent,
                              "narrative": False})["risk"]


# --------------------------------------------------------------------------
# The two gates
# --------------------------------------------------------------------------
def test_negligible_forecast_holds(stub_retriever, quant_findings, sentiment_findings):
    """Gate 0: a move smaller than MIN_MATERIAL_RETURN is not worth trading."""
    quant_findings["forecast_7d_pct"] = 0.05          # 0.05% = 0.0005 < 0.001
    risk = decide(quant_findings, sentiment_findings)
    assert risk["directive"] == "HOLD"
    assert "negligible" in " ".join(risk["reasons"])


def test_low_conviction_holds(stub_retriever, quant_findings, sentiment_findings):
    """Gate 1: a material move below the conviction floor still holds."""
    quant_findings["signal_to_noise"] = config.SIGNAL_TO_NOISE_MIN - 0.01
    risk = decide(quant_findings, sentiment_findings)
    assert risk["directive"] == "HOLD"
    assert risk["tradable"] is False


def test_conflicting_signals_hold_and_flag_a_trap(
        stub_retriever, quant_findings, sentiment_findings):
    """Gate 2: the two evidence streams disagreeing is itself a reason not to act."""
    quant_findings["forecast_7d_pct"] = 1.0                       # up
    sentiment_findings["sentiment_now"] = -config.SENTIMENT_STRONG - 0.1   # bearish
    risk = decide(quant_findings, sentiment_findings)
    assert risk["directive"] == "HOLD"
    assert risk["agreement"] == "conflicting"
    assert "bear trap" in risk["trap_warning"]


# --------------------------------------------------------------------------
# Acting
# --------------------------------------------------------------------------
@pytest.mark.parametrize("forecast_pct, sentiment, expected", [
    (+1.0, +config.SENTIMENT_STRONG + 0.1, "BUY"),    # aligned bullish
    (-1.0, -config.SENTIMENT_STRONG - 0.1, "SELL"),   # aligned bearish
    (+1.0, 0.0, "BUY"),                               # neutral sentiment does not veto
    (-1.0, 0.0, "SELL"),
])
def test_direction_follows_the_forecast(stub_retriever, quant_findings,
                                        sentiment_findings, forecast_pct,
                                        sentiment, expected):
    quant_findings["forecast_7d_pct"] = forecast_pct
    sentiment_findings["sentiment_now"] = sentiment
    risk = decide(quant_findings, sentiment_findings)
    assert risk["directive"] == expected
    assert risk["position_pct"] > 0


def test_agreement_raises_conviction(stub_retriever, quant_findings, sentiment_findings):
    neutral = decide(quant_findings, dict(sentiment_findings, sentiment_now=0.0))
    aligned = decide(quant_findings,
                     dict(sentiment_findings,
                          sentiment_now=config.SENTIMENT_STRONG + 0.1))
    assert aligned["conviction"] > neutral["conviction"]


def test_conviction_saturates_above_twice_the_floor(
        stub_retriever, quant_findings, sentiment_findings):
    """Conviction is capped at 0.95, so the agreement bonus is inert up there.

    Pinned rather than fixed: no traded day in the committed backtest reaches
    2x the floor (observed 1.06-1.74x), so the ceiling never binds in practice.
    If the floor is ever recalibrated downward this test is the warning that the
    sentiment bonus has stopped doing anything.
    """
    quant_findings["signal_to_noise"] = config.SIGNAL_TO_NOISE_MIN * 2.5
    neutral = decide(quant_findings, dict(sentiment_findings, sentiment_now=0.0))
    aligned = decide(quant_findings,
                     dict(sentiment_findings,
                          sentiment_now=config.SENTIMENT_STRONG + 0.1))
    assert neutral["conviction"] == aligned["conviction"] == 0.95


def test_high_volatility_halves_the_position(
        stub_retriever, quant_findings, sentiment_findings):
    normal = decide(quant_findings, sentiment_findings)
    quant_findings["volatility_regime"] = "high"
    high = decide(quant_findings, sentiment_findings)
    assert high["position_pct"] == pytest.approx(round(normal["position_pct"] / 2, 1))


# --------------------------------------------------------------------------
# Regressions for the audit fixes
# --------------------------------------------------------------------------
def test_flat_forecast_is_not_a_sell(stub_retriever, quant_findings, sentiment_findings):
    """B3: a zero forecast used to fall through to the short side.

    It has no direction, so it must not manufacture a SELL-shaped disagreement
    with bullish sentiment.
    """
    quant_findings["forecast_7d_pct"] = 0.0
    sentiment_findings["sentiment_now"] = config.SENTIMENT_STRONG + 0.1
    risk = decide(quant_findings, sentiment_findings)
    assert risk["agreement"] == "no_signal"
    assert risk["directive"] == "HOLD"
    assert risk["trap_warning"] is None


def test_narrative_off_never_calls_the_model(monkeypatch, stub_retriever,
                                             quant_findings, sentiment_findings):
    """B2: the backtest must not reach the language model."""
    from finagent_pulse.agents import committee

    def _boom():
        raise AssertionError("get_writer() called with narrative=False")

    monkeypatch.setattr(committee, "get_writer", _boom)
    risk = decide(quant_findings, sentiment_findings)
    assert risk["directive"] in {"BUY", "SELL", "HOLD"}


# --------------------------------------------------------------------------
# The reproducibility claim
# --------------------------------------------------------------------------
def test_the_rule_is_deterministic(stub_retriever, quant_findings, sentiment_findings):
    a = decide(quant_findings, sentiment_findings)
    b = decide(quant_findings, sentiment_findings)
    for key in ("directive", "position_pct", "conviction", "agreement", "reasons"):
        assert a[key] == b[key]
