"""Ablation study: standard RAG vs Hybrid RAG.

Evaluation protocol
-------------------
There are no human relevance judgements for this corpus, so the benchmark is
built from the corpus itself using a *degraded known-item* protocol, which is
standard practice in IR when editorial judgements are unavailable:

1.  Sample seed headlines that mention at least two financial entities.
2.  Turn each seed into a query that is deliberately *lossy*, so that trivially
    matching the exact string is impossible.  Two query styles are generated:

    ``keyword``   a shuffled subset of the seed's content words -- the way an
                  analyst half-remembers a headline.  Favours lexical search.
    ``semantic``  a natural-language question written from the seed's entities
                  only, sharing almost no surface tokens with the target.
                  Favours dense search.

3.  A document counts as relevant if it shares at least two entities with the
    seed and falls inside a +/-15 day window -- i.e. it belongs to the same
    market event cluster.  The seed itself is always relevant.

Reporting both query styles is the point of the study: a single-retriever
system wins on the style that matches it and collapses on the other, whereas
the fused system should stay strong on both.
"""
from __future__ import annotations

import json
import logging
import random
import re

import numpy as np
import pandas as pd

from finagent_pulse import config
from finagent_pulse.rag.entities import LEXICON
from finagent_pulse.rag.hybrid import MODES, get_retriever

log = logging.getLogger(__name__)

RESULTS_PATH = config.REPORTS / "rag_ablation.json"
RESULTS_CSV = config.REPORTS / "rag_ablation.csv"

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "as", "is", "are",
    "at", "by", "with", "from", "after", "over", "its", "it", "this", "that",
    "be", "was", "were", "has", "have", "will", "may", "amid", "up", "down",
    "new", "more", "than", "but", "not", "what", "how", "why", "s", "t",
}

_QUESTION_TEMPLATES = [
    "What is the market outlook given {a} and {b}?",
    "How are investors reacting to {a} alongside {b}?",
    "What is driving sentiment around {a} and {b}?",
    "Why does {a} matter for {b} right now?",
    "What is the latest news connecting {a} to {b}?",
]


# --------------------------------------------------------------------------
# Benchmark construction
# --------------------------------------------------------------------------
def _content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9&']+", text.lower())
    return [w for w in words if w not in _STOP and len(w) > 2]


def build_benchmark(docs: pd.DataFrame, n_queries: int = 150, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)

    docs = docs.copy()
    docs["entities"] = docs["entities_str"].map(lambda s: s.split("|") if s else [])
    docs["n_ent"] = docs["entities"].str.len()
    pool = docs[docs["n_ent"] >= 2].reset_index(drop=True)
    log.info("Benchmark pool: %d headlines with >=2 entities", len(pool))

    dates = pd.to_datetime(docs["date_str"])
    ent_sets = docs["entities"].map(set).to_numpy()
    all_ids = docs["doc_id"].to_numpy()

    queries: list[dict] = []
    picked = rng.sample(range(len(pool)), min(n_queries, len(pool)))

    for i in picked:
        seed_row = pool.iloc[i]
        seed_ents = set(seed_row["entities"])
        seed_date = pd.Timestamp(seed_row["date_str"])

        # Relevance: same event cluster = >=2 shared entities within +/-15 days.
        window = (dates >= seed_date - pd.Timedelta(days=15)) & \
                 (dates <= seed_date + pd.Timedelta(days=15))
        shared = np.array([len(seed_ents & s) for s in ent_sets])
        rel_mask = window.to_numpy() & (shared >= 2)
        relevant = set(all_ids[rel_mask]) | {seed_row["doc_id"]}

        # Too-large clusters make the task trivially easy; skip them.
        if not (2 <= len(relevant) <= 400):
            continue

        # --- keyword-style query: a lossy subset of the seed's content words
        words = _content_words(seed_row["headline"])
        if len(words) < 4:
            continue
        keep = max(3, int(len(words) * 0.5))
        kw = rng.sample(words, keep)
        rng.shuffle(kw)

        # --- semantic-style query: entity names only, phrased as a question
        names = [LEXICON[e][1][0] for e in sorted(seed_ents) if e in LEXICON][:2]
        if len(names) < 2:
            continue
        sem = rng.choice(_QUESTION_TEMPLATES).format(a=names[0], b=names[1])

        queries.append({
            "seed_id": seed_row["doc_id"],
            "seed_headline": seed_row["headline"],
            "seed_date": seed_row["date_str"],
            "entities": sorted(seed_ents),
            "relevant": sorted(relevant),
            "n_relevant": len(relevant),
            "keyword_query": " ".join(kw),
            "semantic_query": sem,
        })

    log.info("Built %d benchmark queries (median cluster size %d)",
             len(queries), int(np.median([q["n_relevant"] for q in queries])))
    return queries


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _dcg(gains: list[float]) -> float:
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def score_ranking(retrieved_ids: list[str], relevant: set[str], k: int) -> dict:
    """Rank-quality metrics for one query. Read the definitions before comparing.

    Two of these are *not* the textbook forms, because relevance clusters here
    routinely exceed ``k`` and the textbook forms would then be bounded well
    below 1 no matter how good the ranking is:

    ``recall@k``    hits / min(|relevant|, k) -- capped recall. Standard recall
                    (hits / |relevant|) would read ~0.07 for a 140-document
                    cluster at k=10 even for a perfect ranking. Capped recall is
                    therefore higher than standard recall, and the two must never
                    be compared against each other or against published figures.
    ``precision@5`` hits@5 / min(5, |retrieved|) -- divided by what was actually
                    returned, not by 5, so a short result list is not penalised.
                    Identical to standard P@5 whenever 5 or more docs come back,
                    which is the case for every query in this study.

    ``mrr`` and ``ndcg@k`` are standard, with binary gains and an ideal DCG over
    min(|relevant|, k) documents.
    """
    top = retrieved_ids[:k]
    hits = [1.0 if d in relevant else 0.0 for d in top]

    recall = sum(hits) / min(len(relevant), k) if relevant else 0.0
    precision5 = sum(hits[:5]) / min(5, len(top)) if top else 0.0
    rr = next((1.0 / (i + 1) for i, h in enumerate(hits) if h > 0), 0.0)
    ideal = _dcg([1.0] * min(len(relevant), k))
    ndcg = _dcg(hits) / ideal if ideal > 0 else 0.0

    return {"recall@k": recall, "precision@5": precision5, "mrr": rr, "ndcg@k": ndcg}


# --------------------------------------------------------------------------
# Ablation
# --------------------------------------------------------------------------
def run_ablation(n_queries: int = 150, top_k: int = 10, save: bool = True) -> pd.DataFrame:
    retriever = get_retriever()
    queries = build_benchmark(retriever.docs, n_queries=n_queries)

    rows = []
    for style in ("keyword", "semantic"):
        field = f"{style}_query"
        for mode in MODES:
            per_query = []
            for q in queries:
                docs, _ = retriever.retrieve(q[field], mode=mode, top_k=top_k)
                per_query.append(
                    score_ranking([d.doc_id for d in docs], set(q["relevant"]), top_k))
            agg = {m: float(np.mean([p[m] for p in per_query])) for m in per_query[0]}
            rows.append({"query_style": style, "mode": mode,
                         "n_queries": len(queries), **agg})
            log.info("%-9s %-10s recall@%d=%.3f ndcg=%.3f mrr=%.3f",
                     style, mode, top_k, agg["recall@k"], agg["ndcg@k"], agg["mrr"])

    df = pd.DataFrame(rows)

    # Macro-average across both query styles: the headline robustness number.
    overall = (df.groupby("mode")[["recall@k", "precision@5", "mrr", "ndcg@k"]]
               .mean().reset_index())
    overall.insert(0, "query_style", "macro_average")
    overall["n_queries"] = len(queries) * 2
    df = pd.concat([df, overall], ignore_index=True)

    if save:
        df.to_csv(RESULTS_CSV, index=False)
        RESULTS_PATH.write_text(json.dumps({
            "top_k": top_k,
            "n_queries": len(queries),
            "protocol": "degraded known-item; relevant = >=2 shared entities within +/-15 days",
            "results": df.to_dict("records"),
        }, indent=2))
    return df


# --------------------------------------------------------------------------
# Weight calibration (development split)
# --------------------------------------------------------------------------
CALIBRATION_PATH = config.REPORTS / "rag_weight_calibration.csv"

DEV_SEED = 7          # disjoint from the seed used for the reported ablation


def calibrate(n_queries: int = 60, top_k: int = 10,
              bm25_grid: tuple[float, ...] = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0),
              kg_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
              save: bool = True) -> pd.DataFrame:
    """Grid-search the fusion weights on a development split.

    Run on ``DEV_SEED`` queries, which are disjoint from the ``seed=42`` split
    used to report the ablation, so the headline numbers are never tuned on
    the data they are reported against.
    """
    retriever = get_retriever()
    queries = build_benchmark(retriever.docs, n_queries=n_queries, seed=DEV_SEED)

    rows = []
    for w_bm25 in bm25_grid:
        for w_kg in kg_grid:
            per_query = []
            for q in queries:
                for style in ("keyword", "semantic"):
                    docs, _ = retriever.retrieve(
                        q[f"{style}_query"], mode="hybrid_kg", top_k=top_k,
                        bm25_weight=w_bm25, kg_weight=w_kg)
                    per_query.append(score_ranking(
                        [d.doc_id for d in docs], set(q["relevant"]), top_k))
            agg = {m: float(np.mean([p[m] for p in per_query])) for m in per_query[0]}
            rows.append({"bm25_weight": w_bm25, "kg_weight": w_kg, **agg})
            log.info("bm25=%.2f kg=%.2f -> ndcg=%.4f recall=%.4f",
                     w_bm25, w_kg, agg["ndcg@k"], agg["recall@k"])

    df = pd.DataFrame(rows).sort_values("ndcg@k", ascending=False).reset_index(drop=True)
    if save:
        df.to_csv(CALIBRATION_PATH, index=False)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import sys

    if "--calibrate" in sys.argv:
        grid = calibrate()
        print("\n=== Fusion-weight calibration (dev split, seed 7) ===")
        print(grid.round(4).head(10).to_string(index=False))
        best = grid.iloc[0]
        print(f"\nbest: bm25_weight={best.bm25_weight} kg_weight={best.kg_weight}")
    else:
        table = run_ablation()
        print("\n=== Hybrid RAG ablation (top_k=10) ===")
        print(table.round(4).to_string(index=False))
