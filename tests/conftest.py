"""Shared fixtures.

The committee and forecaster modules import torch and chromadb at module level,
so the suite needs the project venv. Nothing here loads a model or an index:
the retriever and the narrator are stubbed, and the price/sentiment frames are
synthetic, so the whole suite runs in seconds and is independent of whether the
pipeline has been run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_features() -> pd.DataFrame:
    """A feature table with the right columns and a reproducible random walk."""
    from finagent_pulse import config

    rng = np.random.default_rng(0)
    # Spans the real study window, so the split tests exercise the actual
    # train_end / val_end boundaries from config rather than invented ones.
    n = 1600
    dates = pd.bdate_range("2018-01-02", periods=n)
    close = 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    df = pd.DataFrame({"date": dates, "close": close})
    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
    df["sent_mean"] = rng.normal(0, 0.15, n)
    for col in config.LSTM.feature_columns:
        if col not in df:
            df[col] = rng.normal(0, 1, n)
    for h in range(1, config.LSTM.horizon + 1):
        df[f"target_h{h}"] = np.log(df["close"]).shift(-h) - np.log(df["close"])
    return df


@pytest.fixture
def stub_retriever(monkeypatch):
    """Replace the ChromaDB-backed retriever with an inert stand-in."""
    from finagent_pulse.agents import committee

    class _Stub:
        def retrieve_principles(self, query, top_k=3):
            return [{"principle": "Margin of safety", "source": "graham",
                     "text": "Buy well below intrinsic value."}]

    monkeypatch.setattr(committee, "get_retriever", lambda: _Stub())
    return _Stub()


@pytest.fixture
def quant_findings() -> dict:
    """A Data Analyst findings dict with knobs the decision rule reads."""
    return {
        "as_of": "2023-06-01",
        "forecast_7d_pct": 1.0,
        # ~1.2x the conviction floor: the median of the traded days in the
        # committed backtest (observed range 1.06-1.74x).
        "signal_to_noise": 0.21,
        "volatility_regime": "normal",
        "horizon_volatility_pct": 2.5,
    }


@pytest.fixture
def sentiment_findings() -> dict:
    return {"sentiment_now": 0.0, "contrarian_flag": None}
