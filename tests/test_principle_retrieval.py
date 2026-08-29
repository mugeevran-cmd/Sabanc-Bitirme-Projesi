"""Which investment principles the Risk Manager cites, and why.

The old code sent one query per decision that ended in a constant
"position sizing and confirmation requirements". That tail dominated the
embedding: the same two principles came back for every decision state and 8 of
the 15 were never retrieved -- including Bull Trap and Bear Trap, which the same
node diagnoses explicitly.

Most of what matters is testable without an index, because the fix is in how the
questions are asked. The one test that does need the real ChromaDB collection is
marked and skips when the pipeline has not been run.
"""
from __future__ import annotations

import pytest

from finagent_pulse import config
from finagent_pulse.agents.committee import (
    principle_queries, retrieve_governing_principles)


def q(agreement="sentiment_neutral", regime="normal", contrarian=None) -> list[str]:
    return principle_queries(agreement, {"volatility_regime": regime},
                             {"contrarian_flag": contrarian})


# --------------------------------------------------------------------------
# Query construction -- the actual fix, no index required
# --------------------------------------------------------------------------
def test_every_state_asks_about_sizing():
    """position_pct rests on it whatever else is true, so it is always asked."""
    for agreement in ("aligned", "conflicting", "sentiment_neutral", "no_signal"):
        for regime in ("low", "normal", "high"):
            assert any("position" in s for s in q(agreement, regime))


def test_the_evidence_situation_reaches_the_query():
    """Each agreement state must ask a different question.

    This is what the old template lost: `agreement` was interpolated into a
    sentence whose constant tail outweighed it.
    """
    asked = {a: q(a)[1] for a in
             ("aligned", "conflicting", "sentiment_neutral", "no_signal")}
    assert len(set(asked.values())) == 4


def test_a_conflict_asks_about_traps():
    assert "trap" in q("conflicting")[1]


def test_high_volatility_changes_the_sizing_question():
    assert q(regime="high")[0] != q(regime="normal")[0]
    assert "volatility" in q(regime="high")[0]


def test_crowding_is_asked_about_only_when_flagged():
    assert len(q(contrarian=None)) == 2
    flagged = q(contrarian="coverage is 80% negative")
    assert len(flagged) == 3
    assert "one-sided" in flagged[-1]


# --------------------------------------------------------------------------
# Merging -- stubbed retriever, no index required
# --------------------------------------------------------------------------
class _FakeRetriever:
    """Returns a distinct principle per query so merge behaviour is visible."""

    def __init__(self):
        self.queries: list[str] = []

    def retrieve_principles(self, query, top_k=3):
        self.queries.append(query)
        n = len(self.queries)
        return [{"principle": f"P{n}.{i}", "source": "test", "text": "..."}
                for i in range(top_k)]


@pytest.fixture
def fake(monkeypatch):
    from finagent_pulse.agents import committee
    r = _FakeRetriever()
    monkeypatch.setattr(committee, "get_retriever", lambda: r)
    return r


def test_each_concern_gets_its_own_retrieval(fake):
    retrieve_governing_principles("conflicting", {"volatility_regime": "high"},
                                  {"contrarian_flag": "one-sided"})
    assert len(fake.queries) == 3


def test_results_are_capped(fake):
    out = retrieve_governing_principles("conflicting", {"volatility_regime": "high"},
                                        {"contrarian_flag": "one-sided"}, cap=4)
    assert len(out) == 4


def test_duplicates_are_dropped(monkeypatch):
    """Two concerns matching the same principle must not cite it twice."""
    from finagent_pulse.agents import committee

    class _Same:
        def retrieve_principles(self, query, top_k=3):
            return [{"principle": "Margin of Safety", "source": "graham", "text": "..."}]

    monkeypatch.setattr(committee, "get_retriever", lambda: _Same())
    out = retrieve_governing_principles("aligned", {"volatility_regime": "low"}, {})
    assert [p["principle"] for p in out] == ["Margin of Safety"]


# --------------------------------------------------------------------------
# The real index -- skipped unless the pipeline has produced one
# --------------------------------------------------------------------------
@pytest.mark.skipif(not any(config.CHROMA_DIR.glob("*")),
                    reason="principles index not built; run the pipeline")
def test_traps_are_actually_retrieved_on_a_conflict():
    """End to end: a conflicting state must surface a trap principle.

    Before the fix this was impossible -- the whole behavioural-finance file was
    unreachable no matter the decision state.
    """
    got = [p["principle"] for p in retrieve_governing_principles(
        "conflicting", {"volatility_regime": "normal"}, {"contrarian_flag": None})]
    assert any("Trap" in name for name in got), got
