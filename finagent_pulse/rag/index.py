"""Builds the three retrieval structures behind the Hybrid RAG system.

1. A dense vector index (ChromaDB + MiniLM sentence embeddings) for semantic
   recall -- finds "equities tumble on tightening fears" from "Fed hawkish".
2. A sparse BM25 index for exact lexical recall -- finds the literal ticker,
   number or proper noun that embeddings routinely blur away.
3. A co-occurrence knowledge graph over financial entities, used to expand a
   query with the concepts that empirically travel with it.

A fourth, much smaller vector collection holds the investment-principles
knowledge base consulted by the Risk Management agent.
"""
from __future__ import annotations

import logging
import pickle
import re

import pandas as pd

from finagent_pulse import config
from finagent_pulse.rag.entities import build_knowledge_graph, extract_entities

log = logging.getLogger(__name__)

DOCS_PATH = config.RAG_INDEX / "documents.parquet"


# --------------------------------------------------------------------------
# BM25 tokenisation
#
# The index and the query path must tokenise identically, so both go through
# ``tokenize()`` and neither is allowed to call ``str.split()`` again.
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9&']+")

# Stamped into the pickled payload and checked on load. A BM25 index built by a
# different tokeniser cannot answer queries tokenised by this one, and the
# failure mode is silent -- every score comes back 0 and the fused retriever
# quietly degrades to vector-only. ``build_bm25_index`` rebuilds on a mismatch
# rather than refusing, so a checkout whose index predates a tokeniser change
# heals itself wherever the corpus is available.
TOKENIZER_VERSION = "v2-regex"


def tokenize(text: str) -> list[str]:
    """Split text into BM25 terms, dropping punctuation.

    Both sides used ``text.lower().split()``, which left punctuation attached to
    the token. An indexed ``"cools,"`` could never match a queried ``"cools"``,
    and because every natural-language benchmark query ends in a question mark,
    one of its two entity terms was always a ``"?"``-suffixed token that matched
    nothing. That inflated the measured collapse of BM25 on semantic queries.

    ``&`` and ``'`` are kept inside tokens so ``s&p`` and ``fed's`` survive as
    single terms.
    """
    return _TOKEN.findall(text.lower())


# --------------------------------------------------------------------------
# Document preparation
# --------------------------------------------------------------------------
def build_documents(force: bool = False) -> pd.DataFrame:
    """Attach entities and market context to every scored headline."""
    if DOCS_PATH.exists() and not force:
        docs = pd.read_parquet(DOCS_PATH)
        docs["entities"] = docs["entities_str"].map(
            lambda s: s.split("|") if s else [])
        return docs

    scored = pd.read_parquet(config.HEADLINES_SCORED)
    docs = scored[["doc_id", "date", "headline", "sentiment", "label",
                   "confidence", "corpus_close"]].copy()
    docs["entities"] = docs["headline"].map(extract_entities)
    docs["entities_str"] = docs["entities"].map("|".join)
    docs["year"] = docs["date"].dt.year
    docs["date_str"] = docs["date"].dt.strftime("%Y-%m-%d")
    # Chroma range filters only accept numbers, so dates are also stored as
    # a sortable YYYYMMDD integer.
    docs["date_int"] = docs["date"].dt.strftime("%Y%m%d").astype(int)

    docs.drop(columns=["entities"]).to_parquet(DOCS_PATH, index=False)
    log.info("Prepared %d documents; %.1f%% carry >=1 entity",
             len(docs), 100 * (docs["entities"].str.len() > 0).mean())
    return docs


# --------------------------------------------------------------------------
# Index builders
# --------------------------------------------------------------------------
def build_vector_index(docs: pd.DataFrame, force: bool = False):
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL)

    existing = {c.name for c in client.list_collections()}
    if config.CHROMA_NEWS_COLLECTION in existing:
        if not force:
            col = client.get_collection(config.CHROMA_NEWS_COLLECTION,
                                        embedding_function=ef)
            log.info("Vector index already built (%d docs)", col.count())
            return col
        client.delete_collection(config.CHROMA_NEWS_COLLECTION)

    col = client.create_collection(
        config.CHROMA_NEWS_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    batch = 1000
    for start in range(0, len(docs), batch):
        chunk = docs.iloc[start:start + batch]
        col.add(
            ids=chunk["doc_id"].tolist(),
            documents=chunk["headline"].tolist(),
            metadatas=[{
                "date": r.date_str,
                "year": int(r.year),
                "date_int": int(r.date_int),
                "sentiment": float(r.sentiment),
                "label": str(r.label),
                "entities": str(r.entities_str),
                "close": float(r.corpus_close),
            } for r in chunk.itertuples()],
        )
        log.info("  embedded %d/%d", min(start + batch, len(docs)), len(docs))

    log.info("Vector index built: %d documents", col.count())
    return col


def build_bm25_index(docs: pd.DataFrame, force: bool = False):
    """Load the BM25 index, rebuilding it if it was not built by ``tokenize``.

    Every caller goes through here, including the retriever. An earlier version
    kept a separate strict loader for the query path, which meant the pipeline
    healed a stale index and the dashboard died on it with a traceback -- the
    same artefact, two behaviours, and the one users met first was the failure.
    """
    from rank_bm25 import BM25Okapi

    if config.BM25_PATH.exists() and not force:
        with open(config.BM25_PATH, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("tokenizer") == TOKENIZER_VERSION:
            return payload
        # A cached index from an older tokeniser is worse than none: it would
        # answer every query with zeros. Rebuild it instead of honouring the
        # cache, so an unpacked artifacts.zip heals itself.
        log.warning("BM25 index was built with tokenizer %r; rebuilding for %r",
                    payload.get("tokenizer", "v1-split"), TOKENIZER_VERSION)

    corpus = [tokenize(h) for h in docs["headline"]]
    payload = {"bm25": BM25Okapi(corpus), "doc_ids": docs["doc_id"].tolist(),
               "tokenizer": TOKENIZER_VERSION}
    with open(config.BM25_PATH, "wb") as fh:
        pickle.dump(payload, fh)
    log.info("BM25 index built over %d documents", len(corpus))
    return payload


def build_kg(docs: pd.DataFrame, force: bool = False):
    if config.KG_PATH.exists() and not force:
        with open(config.KG_PATH, "rb") as fh:
            return pickle.load(fh)

    records = docs[["entities", "sentiment"]].to_dict("records")
    graph = build_knowledge_graph(records)
    with open(config.KG_PATH, "wb") as fh:
        pickle.dump(graph, fh)
    log.info("Knowledge graph: %d nodes, %d edges",
             graph.number_of_nodes(), graph.number_of_edges())
    return graph


def build_principles_index(force: bool = False):
    """Chunk the investment-principles markdown into a small vector collection."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL)

    existing = {c.name for c in client.list_collections()}
    if config.CHROMA_KB_COLLECTION in existing:
        if not force:
            return client.get_collection(config.CHROMA_KB_COLLECTION,
                                         embedding_function=ef)
        client.delete_collection(config.CHROMA_KB_COLLECTION)

    col = client.create_collection(config.CHROMA_KB_COLLECTION, embedding_function=ef)

    ids, texts, metas = [], [], []
    for path in sorted(config.KNOWLEDGE_BASE.glob("*.md")):
        source = path.stem
        # One chunk per '##' section: each principle is self-contained.
        for i, block in enumerate(path.read_text().split("\n## ")):
            block = block.strip()
            if len(block) < 60:
                continue
            lines = block.splitlines()
            title = lines[0].lstrip("# ").strip()
            # The heading is prepended to the body so it is part of what gets
            # embedded -- "Bull Trap" carries most of that chunk's meaning and
            # dropping it would make the chunk much harder to retrieve. It is
            # also stored in metadata, so a caller that renders the name itself
            # would repeat it; HybridRetriever.retrieve_principles strips the
            # prefix back off for exactly that reason.
            body = " ".join(l.strip() for l in lines[1:] if l.strip())
            ids.append(f"{source}-{i}")
            texts.append(f"{title}. {body}" if body else title)
            metas.append({"source": source, "principle": title})

    col.add(ids=ids, documents=texts, metadatas=metas)
    log.info("Principles index: %d chunks from %d files",
             len(ids), len(list(config.KNOWLEDGE_BASE.glob('*.md'))))
    return col


def build_all(force: bool = False) -> dict:
    docs = build_documents(force=force)
    return {
        "documents": docs,
        "vector": build_vector_index(docs, force=force),
        "bm25": build_bm25_index(docs, force=force),
        "kg": build_kg(docs, force=force),
        "principles": build_principles_index(force=force),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    art = build_all(force=True)
    g = art["kg"]
    print(f"\ndocuments : {len(art['documents'])}")
    print(f"kg        : {g.number_of_nodes()} nodes / {g.number_of_edges()} edges")
    top = sorted(g.degree(weight="weight"), key=lambda kv: -kv[1])[:10]
    print("top entities by weighted degree:", top)
