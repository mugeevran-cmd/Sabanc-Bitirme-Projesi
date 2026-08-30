"""The dashboard's projected-vs-realised trajectory table.

Two things are easy to get wrong here and neither would look wrong on screen:
the realised return has to use the same definition and the same denominator as
the projected one or the error column is meaningless, and a forecast whose
horizon runs past the end of the data has to stay seven rows long rather than
silently becoming a shorter forecast.
"""
from __future__ import annotations

import pandas as pd
import pytest

from finagent_pulse import config
from finagent_pulse.app.streamlit_app import _split_of, trajectory_table

H = config.LSTM.horizon


@pytest.fixture
def frame() -> pd.DataFrame:
    """Ten sessions of round numbers, so the arithmetic is checkable by eye."""
    return pd.DataFrame({
        "date": pd.bdate_range("2023-06-01", periods=10),
        "close": [100.0 * (1 + i / 100) for i in range(10)],
    })


@pytest.fixture
def forecast() -> dict:
    """Origin at 100.0, projecting a flat +1% per session."""
    prices = [100.0 * (1 + (i + 1) / 100) for i in range(H)]
    return {
        "origin_close": 100.0,
        "prices": prices,
        "path_return_pct": [(p / 100.0 - 1) * 100 for p in prices],
    }


def test_every_horizon_gets_a_row(frame, forecast):
    t = trajectory_table(frame, frame["date"].iloc[0], forecast)
    assert list(t["Session"]) == [f"t+{i}" for i in range(1, H + 1)]


def test_realised_return_is_measured_against_the_origin_close(frame, forecast):
    """Same denominator as the projection, or the error column means nothing."""
    t = trajectory_table(frame, frame["date"].iloc[0], forecast)
    # close[1] is 101.0 against an origin of 100.0 -> +1.00%
    assert t["Realised return"].iloc[0] == "+1.00%"
    assert t["Realised level"].iloc[0] == "101.00"


def test_error_is_projected_minus_realised(frame, forecast):
    t = trajectory_table(frame, frame["date"].iloc[0], forecast)
    # The fixture projects exactly what happens, so every error is zero.
    assert set(t["Error (pp)"]) == {"+0.00"}


def test_a_wrong_forecast_shows_a_signed_error(frame, forecast):
    forecast["path_return_pct"] = [r + 2.0 for r in forecast["path_return_pct"]]
    t = trajectory_table(frame, frame["date"].iloc[0], forecast)
    assert set(t["Error (pp)"]) == {"+2.00"}


# --------------------------------------------------------------------------
# Horizons that run past the end of the data
# --------------------------------------------------------------------------
def test_a_partly_realised_forecast_keeps_all_its_rows(frame, forecast):
    """Two sessions left: rows 1-2 realised, 3-7 blank, still seven rows."""
    as_of = frame["date"].iloc[-3]
    t = trajectory_table(frame, as_of, forecast)
    assert len(t) == H
    assert (t["Realised level"] != "—").sum() == 2
    assert list(t["Realised level"])[2:] == ["—"] * (H - 2)


def test_the_last_session_realises_nothing(frame, forecast):
    t = trajectory_table(frame, frame["date"].iloc[-1], forecast)
    assert list(t["Realised level"]) == ["—"] * H
    assert list(t["Realised return"]) == ["—"] * H
    assert list(t["Error (pp)"]) == ["—"] * H
    # The projection itself is still shown -- it is a forecast, not a blank row.
    assert all(v != "—" for v in t["Projected level"])


def test_realised_dates_are_the_sessions_that_followed(frame, forecast):
    as_of = frame["date"].iloc[0]
    t = trajectory_table(frame, as_of, forecast)
    assert t["Date"].iloc[0] == frame["date"].iloc[1].date().isoformat()
    assert all(pd.Timestamp(d) > as_of for d in t["Date"] if d != "—")


# --------------------------------------------------------------------------
# The in-sample warning
# --------------------------------------------------------------------------
@pytest.mark.parametrize("date, expected", [
    ("2021-06-15", "training"),
    (config.LSTM.train_end, "training"),
    ("2022-09-01", "validation"),
    (config.LSTM.val_end, "validation"),
    ("2023-06-15", "test"),
])
def test_the_split_is_named_so_in_sample_dates_can_be_flagged(date, expected):
    """A train/val date's agreement is a fit, not forecast accuracy."""
    assert _split_of(pd.Timestamp(date)) == expected
