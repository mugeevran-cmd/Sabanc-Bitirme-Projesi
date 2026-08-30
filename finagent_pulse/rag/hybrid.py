"""Hybrid retrieval: dense + sparse + knowledge-graph expansion.

Four retrieval modes are exposed so the ablation study can isolate exactly what
each component contributes:

``vector``     dense semantic search only (ChromaDB / MiniLM)
``bm25``       sparse lexical search only (BM25-Okapi)
``hybrid``     reciprocal-rank fusion of the two
``hybrid_kg``  fusion, with the query first expanded via the knowledge graph

Reciprocal Rank Fusion is used rather than score interpolation because cosine
similarities and BM25 scores live on incomparable scales; RRF only needs the
ranks, so no per-query normalisation is required.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass

import pandas as pd

from finagent_pulse import config
from finagent_pulse.rag import index as rag_index
from finagent_pulse.rag.entities import expand_query

log = logging.getLogger(__name__)

MODES = ("vector", "bm25", "hybrid", "hybrid_kg")


@dataclass
class RetrievedDoc:
    doc_id: str
    headline: str
    date: str
    sentiment: float
    label: str
    entities: list[str]
    score: float
    rank: int
    provenance: str          # which retriever(s) surfaced this document

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "headline": self.headline, "date": self.date,
            "sentiment": round(self.sentiment, 4), "label": self.label,
            "entities": self.entities, "score": round(self.score, 5),
            "rank": self.rank, "provenance": self.provenance,
        }


class HybridRetriever:
    """Loads the prebuilt indexes and serves the four retrieval modes."""

    def __init__(self) -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        self.docs = rag_index.build_documents()
        self._by_id = self.docs.set_index("doc_id")

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = client.get_collection(
            config.CHROMA_NEWS_COLLECTION, embedding_function=ef)
        self.principles = client.get_collection(
            config.CHROMA_KB_COLLECTION, embedding_function=ef)

        # Refuses an index built by a different tokeniser rather than scoring
        # every document 0 -- see rag_index.load_bm25.
        payload = rag_index.load_bm25()
        self.bm25 = payload["bm25"]
        self.bm25_ids = payload["doc_ids"]

        with open(config.KG_PATH, "rb") as fh:
            self.kg = pickle.load(fh)

        log.info("HybridRetriever ready: %d documents, KG %d nodes",
                 len(self.docs), self.kg.number_of_nodes())

    # ---------------------------------------------------------------- filters
    def _date_filter(self, start: str | None, end: str | None) -> dict | None:
        to_int = lambda d: int(str(d)[:10].replace("-", ""))
        clauses = []
        if start:
            clauses.append({"date_int": {"$gte": to_int(start)}})
        if end:
            clauses.append({"date_int": {"$lte": to_int(end)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    # ------------------------------------------------------------- retrievers
    def _vector_ids(self, query: str, k: int, where: dict | None) -> list[str]:
        res = self.collection.query(query_texts=[query], n_results=k, where=where)
        return list(res["ids"][0])

    def _bm25_ids(self, query: str, k: int,
                  allowed: set[str] | None) -> tuple[list[str], float]:
        """Return the top-k ids *and* the best BM25 score.

        The top score measures how much genuine lexical overlap the query has
        with the corpus at all, and is what drives adaptive fusion weighting.
        """
        # Same tokeniser the index was built with; see rag_index.tokenize.
        scores = self.bm25.get_scores(rag_index.tokenize(query))
        order = scores.argsort()[::-1]
        out: list[str] = []
        for i in order:
            if scores[i] <= 0:
                break
            doc_id = self.bm25_ids[i]
            if allowed is None or doc_id in allowed:
                out.append(doc_id)
                if len(out) >= k:
                    break
        return out, float(scores.max()) if len(scores) else 0.0

    @staticmethod
    def _rrf(rankings: dict[str, list[str]],
             weights: dict[str, float] | None = None,
             k: int = config.RRF_K) -> list[tuple[str, float, str]]:
        """Weighted reciprocal-rank fusion of several ranked id lists.

        Ranks, not scores, are fused because cosine similarity and BM25 live on
        incomparable scales.  Per-arm weights let a retriever that clearly has
        nothing useful to say be discounted instead of dragging the fusion down.
        """
        weights = weights or {}
        fused: dict[str, float] = {}
        origin: dict[str, list[str]] = {}
        for name, ids in rankings.items():
            w = weights.get(name, 1.0)
            if w <= 0:
                continue
            for rank, doc_id in enumerate(ids):
                fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank + 1)
                origin.setdefault(doc_id, []).append(name)
        return [(d, sc, "+".join(sorted(set(origin[d]))))
                for d, sc in sorted(fused.items(), key=lambda kv: -kv[1])]

    # ------------------------------------------------------------------ query
    def retrieve(self,
                 query: str,
                 mode: str = "hybrid_kg",
                 top_k: int = config.RETRIEVAL_TOP_K,
                 start: str | None = None,
                 end: str | None = None,
                 bm25_weight: float | None = None,
                 kg_weight: float | None = None) -> tuple[list[RetrievedDoc], dict]:
        """Retrieve ``top_k`` headlines. Returns ``(docs, diagnostics)``.

        ``bm25_weight`` and ``kg_weight`` override the calibrated fusion
        weights; they exist so the calibration sweep can explore the weight
        space without mutating global configuration.
        """
        w_bm25 = config.BM25_WEIGHT if bm25_weight is None else bm25_weight
        w_kg = config.KG_ARM_WEIGHT if kg_weight is None else kg_weight
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        where = self._date_filter(start, end)
        allowed: set[str] | None = None
        if where is not None:
            mask = pd.Series(True, index=self.docs.index)
            if start:
                mask &= self.docs["date_str"] >= str(start)[:10]
            if end:
                mask &= self.docs["date_str"] <= str(end)[:10]
            allowed = set(self.docs.loc[mask, "doc_id"])

        cand = config.CANDIDATE_K
        neighbours: list[str] = []
        expanded_query = query
        weights: dict[str, float] = {}

        if mode == "vector":
            ids = self._vector_ids(query, top_k, where)
            ranked = [(d, 1.0 / (i + 1), "vector") for i, d in enumerate(ids)]

        elif mode == "bm25":
            ids, _top = self._bm25_ids(query, top_k, allowed)
            ranked = [(d, 1.0 / (i + 1), "bm25") for i, d in enumerate(ids)]

        else:
            bm_ids, _bm_top = self._bm25_ids(query, cand, allowed)
            rankings = {"vector": self._vector_ids(query, cand, where), "bm25": bm_ids}
            # The dense arm is the anchor (weight 1.0); the lexical arm carries
            # a calibrated weight because on natural-language questions BM25
            # ranks close to noise and would otherwise drag the fusion down.
            weights = {"vector": 1.0, "bm25": w_bm25}

            if mode == "hybrid_kg":
                expanded_query, neighbours = expand_query(
                    query, self.kg, top_n=config.KG_EXPANSION_TERMS)
                if neighbours:
                    # The expansion runs as *additional* retrieval arms rather
                    # than replacing the query. Overwriting the query text with
                    # extra terms dilutes both retrievers and loses the user's
                    # original intent; running it alongside adds recall on the
                    # concepts that travel with the query while the original
                    # arms keep precision. Expanded arms are down-weighted
                    # because they answer a broader question than the one asked.
                    exp_bm_ids, _ = self._bm25_ids(expanded_query, cand, allowed)
                    rankings["vector_kg"] = self._vector_ids(expanded_query, cand, where)
                    rankings["bm25_kg"] = exp_bm_ids
                    weights["vector_kg"] = w_kg
                    weights["bm25_kg"] = w_kg * w_bm25

            ranked = self._rrf(rankings, weights=weights)

        out: list[RetrievedDoc] = []
        for rank, (doc_id, score, prov) in enumerate(ranked[:top_k], start=1):
            if doc_id not in self._by_id.index:
                continue
            row = self._by_id.loc[doc_id]
            ents = row["entities_str"].split("|") if row["entities_str"] else []
            out.append(RetrievedDoc(
                doc_id=doc_id, headline=row["headline"], date=row["date_str"],
                sentiment=float(row["sentiment"]), label=str(row["label"]),
                entities=ents, score=float(score), rank=rank, provenance=prov,
            ))

        diagnostics = {
            "mode": mode,
            "original_query": query,
            "expanded_query": expanded_query,
            "kg_neighbours": neighbours,
            "arm_weights": {k: round(v, 3) for k, v in weights.items()},
            "n_returned": len(out),
            "date_window": [start, end],
        }
        return out, diagnostics

    # ------------------------------------------------------- principles lookup
    @staticmethod
    def _principle_body(title: str | None, document: str) -> str:
        """Return the chunk text without the heading the indexer prepended.

        ``rag_index.build_principles_index`` embeds ``"{title}. {body}"`` so the
        principle name is part of the vector. Every caller also renders the name
        from metadata, so returning the raw document makes each citation read
        "**Bull Trap** — Bull Trap. A rally that ...". Strip it here, once, so no
        consumer has to know how the chunk was assembled.
        """
        prefix = f"{title}. "
        if title and document.startswith(prefix):
            return document[len(prefix):].strip()
        # A heading-only chunk has no body to show beyond the name itself.
        return "" if title and document.strip() == title else document.strip()

    def retrieve_principles(self, query: str, top_k: int = 4) -> list[dict]:
        """Semantic lookup over the investment-guideline knowledge base."""
        res = self.principles.query(query_texts=[query], n_results=top_k)
        return [
            {"principle": m.get("principle"), "source": m.get("source"),
             "text": self._principle_body(m.get("principle"), d)}
            for d, m in zip(res["documents"][0], res["metadatas"][0])
        ]


_SINGLETON: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Process-wide singleton -- loading the embedding model is expensive."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = HybridRetriever()
    return _SINGLETON


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    r = get_retriever()
    q = "Is the Federal Reserve about to cut interest rates?"
    for mode in MODES:
        docs, diag = r.retrieve(q, mode=mode, top_k=4, start="2023-01-01", end="2023-12-31")
        print(f"\n=== {mode} ===  expanded={diag['expanded_query'][:80]!r}")
        if diag["kg_neighbours"]:
            print(f"    KG neighbours: {diag['kg_neighbours']}")
        for d in docs:
            print(f"  [{d.date}] ({d.provenance:<12}) {d.sentiment:+.2f}  {d.headline[:78]}")
