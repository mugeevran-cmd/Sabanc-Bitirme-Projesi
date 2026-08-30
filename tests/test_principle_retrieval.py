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


# --------------------------------------------------------------------------
# How a retrieved principle is cited -- no index required
#
# Both defects below were visible in the committed executive report and in the
# dashboard's Investment Committee tab: every bullet read
# "**Margin of Safety** — Margin of Safety. The central concept ... that being..."
# -- the name twice, and the paragraph severed mid-sentence.
# --------------------------------------------------------------------------
BODY = ("A sharp upward move during a structural downtrend that draws in "
        "buyers before resuming the decline.")


def test_the_cited_text_does_not_repeat_the_principle_name():
    """The indexer embeds "{title}. {body}"; the caller renders the title too."""
    from finagent_pulse.rag.hybrid import HybridRetriever

    assert HybridRetriever._principle_body("Bull Trap", f"Bull Trap. {BODY}") == BODY


def test_a_heading_only_chunk_cites_nothing_beyond_its_name():
    from finagent_pulse.rag.hybrid import HybridRetriever

    assert HybridRetriever._principle_body("Bull Trap", "Bull Trap") == ""


def test_text_without_the_prefix_is_left_alone():
    from finagent_pulse.rag.hybrid import HybridRetriever

    assert HybridRetriever._principle_body("Bull Trap", BODY) == BODY
    assert HybridRetriever._principle_body(None, BODY) == BODY


def _risk_findings(principles: list[dict]) -> dict:
    return {"directive": "HOLD", "position_pct": 0.0, "conviction": 0.5,
            "agreement": "conflicting", "trap_warning": None,
            "reasons": ["the two evidence streams disagree"],
            "principles": principles,
            "invalidation": {"horizon_days": 7, "expected_move_pct": 0.3,
                             "noise_band_pct": 2.0}}


def test_principles_are_cited_in_full():
    """Every chunk is one short paragraph, so nothing is truncated.

    The old renderer cut at 200 characters, which severed 11 of the 15
    principles mid-sentence -- including every one whose conclusion is the part
    that justifies the directive it was cited for.
    """
    from finagent_pulse.agents.committee import _render_risk_report

    md = _render_risk_report(
        _risk_findings([{"principle": "Bull Trap", "source": "behavioral_finance",
                         "text": BODY}]), {}, {})
    assert f"- **Bull Trap** — {BODY}" in md
    assert "..." not in md


def test_a_bodyless_principle_renders_as_just_its_name():
    from finagent_pulse.agents.committee import _render_risk_report

    md = _render_risk_report(
        _risk_findings([{"principle": "Bull Trap", "source": "behavioral_finance",
                         "text": ""}]), {}, {})
    assert "- **Bull Trap**\n" in md or md.rstrip().endswith("- **Bull Trap**")
