"""BM25 tokenisation: the index and the query path must agree.

Both sides used `text.lower().split()`, which leaves punctuation glued to the
token. An indexed "cools," never matched a queried "cools", and because every
natural-language benchmark query ends in a question mark, one of its two entity
terms was always a "?"-suffixed token that matched nothing at all. That is a
silent failure -- BM25 returns zeros, the fused retriever degrades to
vector-only, and the ablation reads it as "BM25 collapses on semantic queries".

These tests pin the tokeniser on both sides and pin the version guard that stops
a cached index from an older tokeniser being honoured.
"""
from __future__ import annotations

import pickle

import pytest
from rank_bm25 import BM25Okapi

from finagent_pulse.rag.index import TOKENIZER_VERSION, load_bm25, tokenize

HEADLINES = [
    "Fed signals rate cut as inflation cools, stocks rally",
    "Nvidia earnings beat sends Nasdaq to a record high.",
    "S&P 500 slips; investors await the jobs report",
    "Oil prices tumble after OPEC meeting",
]


# --------------------------------------------------------------------------
# The tokeniser itself
# --------------------------------------------------------------------------
def test_punctuation_is_stripped():
    assert tokenize("inflation cools, stocks rally") == [
        "inflation", "cools", "stocks", "rally"]


def test_a_trailing_question_mark_does_not_change_the_term():
    """Every semantic benchmark query ends in one, on its final entity term."""
    assert tokenize("driving sentiment around inflation?")[-1] == "inflation"


def test_ampersands_and_apostrophes_stay_inside_the_token():
    """`s&p` and `fed's` are single terms, not three and two."""
    assert tokenize("S&P 500 and the Fed's stance") == [
        "s&p", "500", "and", "the", "fed's", "stance"]


def test_short_terms_survive():
    """No length filter: `ai`, `oil` and `fed` are all real query terms here."""
    assert tokenize("AI and oil") == ["ai", "and", "oil"]


# --------------------------------------------------------------------------
# The property that was actually broken
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bm25() -> BM25Okapi:
    """An index built exactly the way build_bm25_index builds it."""
    return BM25Okapi([tokenize(h) for h in HEADLINES])


def test_a_term_followed_by_a_comma_in_the_corpus_is_retrievable(bm25):
    """"cools," in the headline must match "cools" in the query."""
    scores = bm25.get_scores(tokenize("cools"))
    assert scores[0] > 0


def test_a_natural_language_question_reaches_the_corpus(bm25):
    """The regression: under `.split()` every one of these scored exactly 0."""
    scores = bm25.get_scores(
        tokenize("What is driving sentiment around oil and inflation?"))
    assert scores.max() > 0
    # Both the "?"-terminated term and the mid-sentence one must contribute.
    assert scores[0] > 0 and scores[3] > 0


def test_the_old_query_tokenisation_loses_the_question_marked_term(bm25):
    """Spells out the query-side defect, so a revert cannot pass quietly.

    Headline 0 is reachable only through "inflation", which the naive split
    leaves as "inflation?" -- so the whole document drops out of the result.
    """
    question = "What is driving sentiment around oil and inflation?"
    assert bm25.get_scores(question.lower().split())[0] == 0.0
    assert bm25.get_scores(tokenize(question))[0] > 0


def test_the_old_index_tokenisation_hides_terms_behind_punctuation():
    """And the index-side defect: "cools," is unreachable from any query."""
    naive_index = BM25Okapi([h.lower().split() for h in HEADLINES])
    assert naive_index.get_scores(tokenize("cools"))[0] == 0.0
    assert BM25Okapi([tokenize(h) for h in HEADLINES]
                     ).get_scores(tokenize("cools"))[0] > 0


# --------------------------------------------------------------------------
# The version guard
# --------------------------------------------------------------------------
def test_a_stale_index_is_refused_not_silently_used(tmp_path):
    """An artifacts.zip built before this fix must raise, not score zeros."""
    path = tmp_path / "bm25.pkl"
    path.write_bytes(pickle.dumps({"bm25": None, "doc_ids": []}))   # no version
    with pytest.raises(RuntimeError, match="v1-split"):
        load_bm25(path)


def test_a_current_index_loads(tmp_path):
    path = tmp_path / "bm25.pkl"
    path.write_bytes(pickle.dumps(
        {"bm25": None, "doc_ids": ["h0"], "tokenizer": TOKENIZER_VERSION}))
    assert load_bm25(path)["doc_ids"] == ["h0"]


def test_the_error_says_how_to_rebuild(tmp_path):
    path = tmp_path / "bm25.pkl"
    path.write_bytes(pickle.dumps({"bm25": None, "doc_ids": [],
                                   "tokenizer": "something-else"}))
    with pytest.raises(RuntimeError, match=r"--only rag --force"):
        load_bm25(path)
