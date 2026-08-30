"""Walk-forward folds: the split arithmetic and the directional baseline.

Fitting four models takes minutes, so what is tested here is the part that can
silently go wrong without anyone noticing: whether each fold's test window is
actually bounded to its own regime, and whether the directional edge is measured
against the right baseline. A fold that quietly tested on everything after its
validation cut would mix a bear market into a bull one and no metric would say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finagent_pulse import walkforward
from finagent_pulse.models.forecaster import build_windows, checkpoint_paths, split_windows


# --------------------------------------------------------------------------
# Fold definitions
# --------------------------------------------------------------------------
def test_folds_are_chronological_and_non_overlapping():
    """Each fold must train before it validates and validate before it tests."""
    for name, train_end, val_end, test_end, _ in walkforward.FOLDS:
        assert pd.Timestamp(train_end) < pd.Timestamp(val_end), name
        if test_end is not None:
            assert pd.Timestamp(val_end) < pd.Timestamp(test_end), name


def test_the_folds_cover_distinct_regimes():
    windows = [(f[1], f[3]) for f in walkforward.FOLDS]
    assert len(set(windows)) == len(windows)


def test_a_bear_market_fold_is_present():
    """The point of the exercise. 2022 fell 20%; without it nothing is answered."""
    names = [f[0] for f in walkforward.FOLDS]
    assert "bear_2022" in names
    bear = next(f for f in walkforward.FOLDS if f[0] == "bear_2022")
    assert bear[3].startswith("2022")


# --------------------------------------------------------------------------
# test_end actually bounds the window
# --------------------------------------------------------------------------
def test_test_end_bounds_the_test_split(synthetic_features):
    """Without this a 2022 fold would also be scored on 2023 and 2024."""
    cfg = walkforward.fold_config("2021-06-30", "2021-12-31", "2022-12-31")
    _X, _y, origins, _c = build_windows(synthetic_features, cfg.feature_columns, cfg)
    _tr, _va, te = split_windows(origins, cfg)
    assert te.sum() > 0
    assert origins[te].max() <= pd.Timestamp("2022-12-31")
    assert origins[te].min() > pd.Timestamp("2021-12-31")


def test_no_test_end_keeps_the_shipped_behaviour(synthetic_features):
    cfg = walkforward.fold_config("2021-06-30", "2021-12-31", None)
    _X, _y, origins, _c = build_windows(synthetic_features, cfg.feature_columns, cfg)
    _tr, _va, te = split_windows(origins, cfg)
    assert origins[te].max() == origins.max()


def test_bounding_the_test_window_never_touches_train_or_val(synthetic_features):
    unbounded = walkforward.fold_config("2021-06-30", "2021-12-31", None)
    bounded = walkforward.fold_config("2021-06-30", "2021-12-31", "2022-12-31")
    _X, _y, origins, _c = build_windows(synthetic_features, unbounded.feature_columns,
                                        unbounded)
    a_tr, a_va, _ = split_windows(origins, unbounded)
    b_tr, b_va, _ = split_windows(origins, bounded)
    assert (a_tr == b_tr).all()
    assert (a_va == b_va).all()


# --------------------------------------------------------------------------
# Artefact paths -- a fold must not overwrite the shipped model's outputs
# --------------------------------------------------------------------------
def test_a_fold_writes_beside_its_own_checkpoint(tmp_path):
    ckpt, scaler, metrics, preds = checkpoint_paths(tmp_path / "walkforward_bear.pt")
    for path in (scaler, metrics, preds):
        assert path.parent == tmp_path
        assert "walkforward_bear" in path.name


def test_the_default_paths_are_the_shipped_ones():
    from finagent_pulse.models import forecaster
    assert checkpoint_paths(None) == (forecaster.CKPT, forecaster.SCALER_PATH,
                                      forecaster.METRICS_PATH, forecaster.PREDICTIONS_PATH)


# --------------------------------------------------------------------------
# The baseline that matters
# --------------------------------------------------------------------------
def test_a_permanent_bull_call_scores_no_edge():
    """A model that always predicts "up" must measure as adding nothing.

    Directional accuracy alone would report 70% here and look like skill.
    """
    realised = [1.0] * 70 + [-1.0] * 30
    decisions = pd.DataFrame({"forecast_7d_pct": [0.5] * 100,
                              "realised_7d_pct": realised})
    edge = walkforward.directional_edge(decisions)
    assert edge["decision_day_accuracy"] == pytest.approx(0.70)
    assert edge["always_up_accuracy"] == pytest.approx(0.70)
    assert edge["edge_over_always_up"] == pytest.approx(0.0)
    assert edge["share_forecast_negative"] == 0.0


def test_a_model_that_calls_the_turns_scores_an_edge():
    rng = np.random.default_rng(0)
    realised = rng.normal(0.2, 1.0, 200)
    decisions = pd.DataFrame({"forecast_7d_pct": realised,      # perfect foresight
                              "realised_7d_pct": realised})
    edge = walkforward.directional_edge(decisions)
    assert edge["decision_day_accuracy"] == pytest.approx(1.0)
    assert edge["edge_over_always_up"] > 0.2


def test_the_forecast_bias_is_reported():
    decisions = pd.DataFrame({"forecast_7d_pct": [2.0] * 50,
                              "realised_7d_pct": [-0.5] * 50})
    edge = walkforward.directional_edge(decisions)
    assert edge["mean_forecast_pct"] == pytest.approx(2.0)
    assert edge["mean_realised_pct"] == pytest.approx(-0.5)
    assert edge["share_forecast_negative"] == 0.0
