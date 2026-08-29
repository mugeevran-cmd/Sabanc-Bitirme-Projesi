"""The tuning split and the reported split must not share documents.

The report states that the fusion weights are grid-searched on a development
split "disjoint from the reported benchmark". That was asserted in prose and not
enforced: both splits drew from the same pool with different random seeds, which
overlaps by a couple of documents on average -- and, worse, the +/-15 day
relevance clusters mean a dev query and an eval query can share almost all of
their relevant sets even when the seed documents differ.

Disjointness is now a property of the code, so it is testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finagent_pulse.rag.evaluate import DEV_SHARE, build_benchmark, partition_of


@pytest.fixture(scope="module")
def fake_docs() -> pd.DataFrame:
    """A document table with the columns build_benchmark reads."""
    rng = np.random.default_rng(0)
    n = 2000
    ents = ["FED", "INFLATION", "RATES", "SP500", "OIL", "EARNINGS", "NVDA"]
    dates = pd.bdate_range("2020-01-01", periods=n)
    rows = []
    for i in range(n):
        picked = sorted(rng.choice(ents, size=rng.integers(2, 4), replace=False))
        rows.append({
            "doc_id": f"h{i:06d}",
            "date_str": dates[i].strftime("%Y-%m-%d"),
            "entities_str": "|".join(picked),
            "headline": f"Markets react as {' and '.join(picked)} shift again in session {i}",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The partition itself
# --------------------------------------------------------------------------
def test_a_document_is_always_in_the_same_partition():
    assert partition_of("h000123") == partition_of("h000123")


def test_the_partition_does_not_depend_on_position():
    """Assignment is by hash of the id, so re-indexing cannot move a document.

    Row-parity or slicing would silently reassign documents whenever the corpus
    grows, quietly leaking last run's tuning set into this run's evaluation.
    """
    ids = [f"h{i:06d}" for i in range(500)]
    first = {i: partition_of(i) for i in ids}
    second = {i: partition_of(i) for i in reversed(ids)}
    assert first == second


def test_the_split_is_roughly_the_configured_share():
    ids = [f"h{i:06d}" for i in range(5000)]
    dev = sum(partition_of(i) == "dev" for i in ids) / len(ids)
    assert DEV_SHARE - 0.03 < dev < DEV_SHARE + 0.03


def test_every_document_lands_somewhere():
    assert {partition_of(f"h{i:06d}") for i in range(200)} <= {"dev", "eval"}


# --------------------------------------------------------------------------
# The benchmarks built on top of it
# --------------------------------------------------------------------------
def test_dev_and_eval_benchmarks_share_no_seed(fake_docs):
    """The property the report claims, now actually enforced."""
    dev = build_benchmark(fake_docs, n_queries=60, seed=7, split="dev")
    ev = build_benchmark(fake_docs, n_queries=150, seed=42, split="eval")
    assert dev and ev
    assert not ({q["seed_id"] for q in dev} & {q["seed_id"] for q in ev})


def test_the_seed_partitions_disagree_about_every_document(fake_docs):
    dev = build_benchmark(fake_docs, n_queries=60, seed=7, split="dev")
    for q in dev:
        assert partition_of(q["seed_id"]) == "dev"


def test_an_unknown_split_is_rejected(fake_docs):
    with pytest.raises(ValueError, match="dev.*eval"):
        build_benchmark(fake_docs, split="test")


def test_the_benchmark_is_reproducible(fake_docs):
    a = build_benchmark(fake_docs, n_queries=40, seed=42, split="eval")
    b = build_benchmark(fake_docs, n_queries=40, seed=42, split="eval")
    assert [q["seed_id"] for q in a] == [q["seed_id"] for q in b]
