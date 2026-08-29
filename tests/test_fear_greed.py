"""The Fear & Greed index: one definition, computed point-in-time.

Before the fix this index existed in two places with two different formulas,
both ranked against the whole 2018-2024 series. On 2024-03-04 the dashboard
read 88/100 "Extreme Greed" while the executive report read 32/100 "Fear" for
the same date and the same data. These tests exist so that cannot come back.
"""
from __future__ import annotations

import pytest

from finagent_pulse.models.sentiment import (
    FEAR_GREED_MIN_HISTORY, fear_greed_at, fear_greed_index)


def test_series_and_scalar_agree(synthetic_features):
    """One definition: the series form and the single-date form must match.

    These are the two entry points the dashboard and the committee use. They
    disagreeing is exactly the bug that was fixed.
    """
    series = fear_greed_index(synthetic_features)
    for i in (200, 600, 1000, len(synthetic_features) - 1):
        scalar = fear_greed_at(synthetic_features.iloc[:i + 1])
        assert scalar == pytest.approx(float(series.iloc[i]), abs=0.051)


def test_a_past_value_does_not_move_when_the_future_arrives(synthetic_features):
    """No lookahead: what a date reads must be computable on that date."""
    cut = 800
    value_then = fear_greed_at(synthetic_features.iloc[:cut + 1])
    series_now = fear_greed_index(synthetic_features)
    assert value_then == pytest.approx(float(series_now.iloc[cut]), abs=0.051)

    # And extending the history must leave the earlier reading untouched.
    longer = fear_greed_index(synthetic_features.iloc[:cut + 300])
    assert float(longer.iloc[cut]) == pytest.approx(
        float(series_now.iloc[cut]), abs=0.051)


def test_the_scale_is_a_percentile(synthetic_features):
    series = fear_greed_index(synthetic_features).dropna()
    assert series.min() >= 0.0
    assert series.max() <= 100.0
    # The final session is ranked against everything, so it can reach the ends.
    assert 0.0 <= fear_greed_at(synthetic_features) <= 100.0


def test_warm_up_is_neutral_not_wrong(synthetic_features):
    """Too little history to rank against returns 50, never a fabricated extreme."""
    assert fear_greed_at(synthetic_features.iloc[:5]) == 50.0
    assert fear_greed_at(synthetic_features.iloc[:FEAR_GREED_MIN_HISTORY]) == 50.0


def test_smoothing_is_applied(synthetic_features):
    """The index ranks 20-day smoothed sentiment, not the raw daily value.

    The committee's old formula skipped the smoothing, which is why its reading
    jumped ~29 points between consecutive sessions.
    """
    series = fear_greed_index(synthetic_features).dropna()
    raw_rank = (synthetic_features["sent_mean"].rank(pct=True) * 100)
    assert series.diff().abs().mean() < raw_rank.diff().abs().mean() / 2
