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


# --------------------------------------------------------------------------
# The committed reports must not fall behind the code that writes them
#
# `_block_bootstrap_p` was added and the report was updated to quote its
# p-values, but the evaluate stage was never re-run: for two commits the
# committed JSON carried only the i.i.d. Pearson p while the write-up cited a
# block-bootstrap figure that existed nowhere in the repository. Nothing failed.
# --------------------------------------------------------------------------
def _committed(name: str) -> dict:
    path = config.REPORTS / name
    if not path.exists():
        pytest.skip(f"pipeline has not produced {name}")
    return json.loads(path.read_text())


def test_the_committed_sentiment_report_carries_a_dependence_aware_p():
    """The report quotes it, so it has to be in the artefact that backs it."""
    sv = _committed("sentiment_validation.json")
    for block in ("corr_same_day", "corr_next_day"):
        assert "p_value_block_bootstrap" in sv[block], (
            f"{block} has no block-bootstrap p -- reports/ is stale against "
            "evaluation.py. Re-run: python -m finagent_pulse.pipeline --only evaluate")


def test_the_summary_and_the_sentiment_report_agree():
    """`run_all` writes the same dict to both; drift means one was not re-run."""
    assert _committed("evaluation_summary.json")["sentiment_validation"] == \
        _committed("sentiment_validation.json")


# --------------------------------------------------------------------------
# C2 -- paired comparison between retrieval modes
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def paired():
    from finagent_pulse.rag.evaluate import paired_bootstrap
    return paired_bootstrap


def test_no_difference_is_reported_as_no_difference(paired):
    rng = np.random.default_rng(10)
    scores = rng.random(150)
    out = paired(list(scores), list(scores), n_boot=1000)
    assert out["mean_difference"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)


def test_a_consistent_edge_is_detected(paired):
    """A small but consistent per-query gain should clear significance."""
    rng = np.random.default_rng(11)
    b = rng.random(150)
    a = b + rng.normal(0.02, 0.01, 150)
    out = paired(list(a), list(b), n_boot=2000)
    assert out["mean_difference"] > 0
    assert out["ci_low"] > 0
    assert out["p_value"] < 0.05


def test_a_wash_is_not_detected(paired):
    """Equal-and-opposite per-query effects must not read as a result.

    This is the hybrid_kg case: a keyword-side gain and a semantic-side loss
    that cancel. Pooled, that is not evidence of anything.
    """
    rng = np.random.default_rng(12)
    b = rng.random(200)
    a = b + np.concatenate([rng.normal(+0.05, 0.02, 100),
                            rng.normal(-0.05, 0.02, 100)])
    out = paired(list(a), list(b), n_boot=2000)
    assert out["p_value"] > 0.05
    assert out["ci_low"] < 0 < out["ci_high"]


def test_the_interval_brackets_the_estimate(paired):
    rng = np.random.default_rng(13)
    b = rng.random(120)
    a = b + rng.normal(0.03, 0.05, 120)
    out = paired(list(a), list(b), n_boot=2000)
    assert out["ci_low"] < out["mean_difference"] < out["ci_high"]
    assert out["n_queries"] == 120


def test_paired_bootstrap_is_seeded(paired):
    rng = np.random.default_rng(14)
    a, b = rng.random(100), rng.random(100)
    assert paired(list(a), list(b), n_boot=500) == paired(list(a), list(b), n_boot=500)
