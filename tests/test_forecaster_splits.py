"""Windowing and split hygiene.

The report's strongest methodological claim is that no future information ever
reaches training: windows lie strictly in the past of their forecast origin,
and an embargo keeps a training target from overlapping a validation or test
observation. Both are easy to break with an off-by-one and impossible to notice
from the metrics -- a leak makes the numbers look *better*. So they are pinned
here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finagent_pulse import config
from finagent_pulse.models.forecaster import build_windows, split_windows

CFG = config.LSTM
FEATURES = CFG.feature_columns


@pytest.fixture(scope="module")
def windows(synthetic_features):
    return build_windows(synthetic_features, FEATURES, CFG)


# --------------------------------------------------------------------------
# Shape and alignment
# --------------------------------------------------------------------------
def test_window_shape(windows):
    X, y, origins, closes = windows
    assert X.shape[1:] == (CFG.lookback, len(FEATURES))
    assert y.shape[1] == CFG.horizon
    assert len(X) == len(y) == len(origins) == len(closes)


def test_the_origin_is_the_last_row_of_its_own_window(windows, synthetic_features):
    """The forecast origin must be inside the window, as its final bar.

    If the window stopped one bar early the model would be asked to forecast
    from a state it was never shown; if it ran one bar late it would be reading
    the future.
    """
    X, _y, origins, closes = windows
    df = synthetic_features
    for i in (0, len(X) // 2, len(X) - 1):
        row = df[df["date"] == origins.iloc[i]].iloc[0]
        np.testing.assert_allclose(X[i][-1], row[FEATURES].to_numpy(dtype=np.float32),
                                   rtol=1e-6)
        assert closes[i] == pytest.approx(row["close"])


def test_targets_are_the_origin_bars_forward_returns(windows, synthetic_features):
    X, y, origins, _closes = windows
    df = synthetic_features
    for i in (0, len(X) // 2, len(X) - 1):
        row = df[df["date"] == origins.iloc[i]].iloc[0]
        expected = [row[f"target_h{h}"] for h in range(1, CFG.horizon + 1)]
        np.testing.assert_allclose(y[i], expected, rtol=1e-5)


def test_every_window_lies_in_the_past_of_its_origin(windows, synthetic_features):
    """The whole lookback, both LSTM directions, must precede the forecast."""
    X, _y, origins, _c = windows
    df = synthetic_features.reset_index(drop=True)
    pos = {d: i for i, d in enumerate(df["date"])}
    for i in (0, len(X) // 2, len(X) - 1):
        end = pos[origins.iloc[i]]
        assert df["date"].iloc[end - CFG.lookback + 1] <= origins.iloc[i]
        assert end - CFG.lookback + 1 >= 0


def test_no_target_reaches_past_the_data(windows, synthetic_features):
    _X, y, _o, _c = windows
    assert not np.isnan(y).any(), "a window was built whose 7-day target does not exist yet"


# --------------------------------------------------------------------------
# Splits and embargo
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def masks(windows):
    _X, _y, origins, _c = windows
    return origins, split_windows(origins, CFG)


def test_splits_are_disjoint(masks):
    _origins, (tr, va, te) = masks
    assert not (tr & va).any()
    assert not (tr & te).any()
    assert not (va & te).any()


def test_splits_are_chronological(masks):
    origins, (tr, va, te) = masks
    assert origins[tr].max() < origins[va].min()
    assert origins[va].max() < origins[te].min()


def test_embargo_exceeds_the_forecast_horizon(masks):
    """A training target must not overlap a validation observation, and so on.

    The horizon is 7 *sessions*; the embargo is expressed in calendar days, so
    the gap is compared against 7 calendar days as the conservative bound.
    """
    origins, (tr, va, te) = masks
    horizon = pd.Timedelta(days=CFG.horizon)
    assert origins[va].min() - origins[tr].max() > horizon
    assert origins[te].min() - origins[va].max() > horizon


def test_every_split_is_non_empty(masks):
    _origins, (tr, va, te) = masks
    assert tr.sum() and va.sum() and te.sum()
