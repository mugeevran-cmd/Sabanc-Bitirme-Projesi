"""Metric definitions and the dependence-aware significance test.

Two of the retrieval metrics are deliberately *not* the textbook forms, and the
correlation p-values are deliberately not `pearsonr`'s. Both choices are easy to
undo by accident, and both would change what the report is entitled to claim.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from finagent_pulse import config
from finagent_pulse.rag.evaluate import score_ranking


# --------------------------------------------------------------------------
# B7 -- the metric definitions the report depends on
# --------------------------------------------------------------------------
def test_recall_is_capped_not_standard():
    """recall@k divides by min(|relevant|, k), so it is NOT standard recall.

    With 20 relevant documents and k=5, a perfect ranking can only retrieve 5 of
    them. Standard recall would cap that at 0.25 no matter how good the system
    is; capped recall reads 1.0. The two are not comparable, which is why the
    docstring says so and why this test spells out the arithmetic.
    """
    relevant = {f"d{i}" for i in range(20)}
    retrieved = ["d0", "x", "d1", "y", "z"]
    m = score_ranking(retrieved, relevant, k=5)

    assert m["recall@k"] == pytest.approx(2 / 5)      # capped: hits / min(20, 5)
    assert m["recall@k"] != pytest.approx(2 / 20)     # standard recall would be 0.10


def test_precision_at_5_divides_by_what_was_returned():
    """A short result list is not penalised -- documented, non-standard."""
    m = score_ranking(["a", "b"], {"a"}, k=5)
    assert m["precision@5"] == pytest.approx(1 / 2)   # not 1/5


def test_mrr_and_ndcg_are_standard():
    relevant = {f"d{i}" for i in range(20)}
    m = score_ranking(["d0", "x", "d1", "y", "z"], relevant, k=5)

    assert m["mrr"] == pytest.approx(1.0)             # first hit at rank 1
    dcg = 1 / np.log2(2) + 1 / np.log2(4)             # hits at ranks 1 and 3
    ideal = sum(1 / np.log2(i + 2) for i in range(5))
    assert m["ndcg@k"] == pytest.approx(dcg / ideal)


def test_a_miss_scores_zero_everywhere():
    m = score_ranking(["x", "y"], {"a", "b"}, k=5)
    assert m["mrr"] == 0.0
    assert m["ndcg@k"] == 0.0
    assert m["recall@k"] == 0.0


def test_empty_retrieval_does_not_raise():
    m = score_ranking([], {"a"}, k=5)
    assert all(v == 0.0 for v in m.values())


# --------------------------------------------------------------------------
# B8 -- autocorrelation-aware p-values
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bootstrap():
    from finagent_pulse.evaluation import _block_bootstrap_p
    return _block_bootstrap_p


def test_the_test_has_roughly_the_right_size(bootstrap):
    """On independent series the p-value should be roughly uniform.

    Asserted across several draws rather than one: for a well-calibrated test a
    single independent pair lands below 0.05 about 5% of the time, so a
    one-draw threshold would be a coin flip dressed up as an assertion.
    """
    rng = np.random.default_rng(1)
    ps = []
    for _ in range(8):
        x, y = rng.normal(size=300), rng.normal(size=300)
        ps.append(bootstrap(x, y, n_boot=400)[1])
    assert sum(p > 0.05 for p in ps) >= 6, f"too many false positives: {ps}"
    assert min(ps) > 0.005, f"an independent pair scored implausibly low: {ps}"


def test_a_real_relationship_survives(bootstrap):
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    y = x + rng.normal(scale=0.3, size=300)
    r, p = bootstrap(x, y, n_boot=400)
    assert r > 0.9
    assert p < 0.05


def test_autocorrelation_is_punished(bootstrap):
    """The whole point: on persistent series the i.i.d. p is far too small."""
    from scipy import stats

    rng = np.random.default_rng(3)
    # Two independent random walks -- correlated by construction, causally unrelated.
    x = np.cumsum(rng.normal(size=300))
    y = np.cumsum(rng.normal(size=300))
    _r, p_block = bootstrap(x, y, n_boot=400)
    _, p_iid = stats.pearsonr(x, y)
    assert p_block > p_iid * 100, "the block bootstrap must be far more conservative"


def test_the_bootstrap_is_seeded(bootstrap):
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=200), rng.normal(size=200)
    assert bootstrap(x, y, n_boot=200) == bootstrap(x, y, n_boot=200)


# --------------------------------------------------------------------------
# B1 -- the calibrated constant and its derivation stay in sync
# --------------------------------------------------------------------------
def test_the_threshold_comes_from_the_calibration_report():
    path = config.REPORTS / "decision_thresholds.json"
    if not path.exists():
        pytest.skip("pipeline has not produced decision_thresholds.json")
    expected = json.loads(path.read_text())["signal_to_noise_min"]
    assert config.SIGNAL_TO_NOISE_MIN == pytest.approx(expected)


def test_a_missing_report_falls_back_to_the_literal(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    assert config._calibrated("signal_to_noise_min", 0.175) == 0.175


def test_an_unknown_key_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    (tmp_path / "decision_thresholds.json").write_text(json.dumps({"other": 1.0}))
    assert config._calibrated("signal_to_noise_min", 0.175) == 0.175
