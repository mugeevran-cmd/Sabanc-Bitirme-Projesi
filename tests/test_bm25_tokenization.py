"""BM25 tokenisation: the index and the query path must agree.

Both sides used `text.lower().split()`, which leaves punctuation glued to the
token. An indexed "cools," never matched a queried "cools", and because every
natural-language benchmark query ends in a question mark, one of its two entity
terms was always a "?"-suffixed token that matched nothing at all. That is a
silent failure -- BM25 returns zeros, the fused retriever degrades to
vector-only, and the ablation reads it as "BM25 collapses on semantic queries".

These tests pin the tokeniser on both sides, and pin the version stamp that
makes an index built by an older tokeniser rebuild itself instead of being
honoured.
"""
from __future__ import annotations

import pickle

import pandas as pd
import pytest
from rank_bm25 import BM25Okapi

from finagent_pulse import config
from finagent_pulse.rag.index import (
    TOKENIZER_VERSION, build_bm25_index, tokenize)

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
# The version stamp
#
# Every caller loads the index through build_bm25_index, so a checkout whose
# pickle predates a tokeniser change heals itself. A previous version kept a
# separate strict loader on the query path, which meant the pipeline rebuilt a
# stale index and the dashboard died on it -- one artefact, two behaviours.
# --------------------------------------------------------------------------
@pytest.fixture
def docs() -> pd.DataFrame:
    return pd.DataFrame({"doc_id": [f"h{i}" for i in range(len(HEADLINES))],
                         "headline": HEADLINES})


@pytest.fixture
def index_path(tmp_path, monkeypatch):
    path = tmp_path / "bm25.pkl"
    monkeypatch.setattr(config, "BM25_PATH", path)
    return path


def test_an_index_from_an_older_tokeniser_is_rebuilt(index_path, docs):
    """What an unpacked artifacts.zip looks like after a tokeniser change."""
    index_path.write_bytes(pickle.dumps({"bm25": None, "doc_ids": []}))  # no stamp

    payload = build_bm25_index(docs)

    assert payload["tokenizer"] == TOKENIZER_VERSION
    assert payload["doc_ids"] == list(docs["doc_id"])
    # And the rebuilt index actually answers, rather than scoring everything 0.
    assert payload["bm25"].get_scores(tokenize("inflation cools?")).max() > 0


def test_a_current_index_is_reused_untouched(index_path, docs):
    marker = {"bm25": "not-rebuilt", "doc_ids": ["sentinel"],
              "tokenizer": TOKENIZER_VERSION}
    index_path.write_bytes(pickle.dumps(marker))
    assert build_bm25_index(docs)["doc_ids"] == ["sentinel"]


def test_force_rebuilds_even_a_current_index(index_path, docs):
    index_path.write_bytes(pickle.dumps(
        {"bm25": "not-rebuilt", "doc_ids": ["sentinel"],
         "tokenizer": TOKENIZER_VERSION}))
    assert build_bm25_index(docs, force=True)["doc_ids"] == list(docs["doc_id"])


def test_a_missing_index_is_built_from_scratch(index_path, docs):
    assert not index_path.exists()
    payload = build_bm25_index(docs)
    assert index_path.exists() and payload["tokenizer"] == TOKENIZER_VERSION
