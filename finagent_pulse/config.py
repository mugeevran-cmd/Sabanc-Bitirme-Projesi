"""Central configuration for FinAgent-Pulse.

Every path, hyper-parameter and model id used across the pipeline lives here so
that experiments stay reproducible and the Streamlit app and the batch pipeline
never drift apart.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
DATA_RAW = ROOT / "data_raw"
DATA_PROCESSED = ROOT / "data_processed"
MODELS_OUT = ROOT / "models_out"
RAG_INDEX = ROOT / "rag_index"
REPORTS = ROOT / "reports"
KNOWLEDGE_BASE = ROOT / "finagent_pulse" / "knowledge_base"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS_OUT, RAG_INDEX, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

HEADLINES_CSV = DATA_RAW / "sp500_headlines_2008_2024.csv"
MARKET_CSV = DATA_RAW / "sp500_market.csv"

HEADLINES_CLEAN = DATA_PROCESSED / "headlines_clean.parquet"
HEADLINES_SCORED = DATA_PROCESSED / "headlines_scored.parquet"
DAILY_SENTIMENT = DATA_PROCESSED / "daily_sentiment.parquet"
FEATURES = DATA_PROCESSED / "features.parquet"

# --------------------------------------------------------------------------
# Data scope -- the proposal restricts the study window to 2018-2024.
# The Kaggle corpus itself stops on 2024-03-04, which becomes the hard end.
# --------------------------------------------------------------------------
KAGGLE_DATASET = "dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024"
KAGGLE_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
TICKER = "^GSPC"          # S&P 500 index
TICKER_LABEL = "S&P 500"

# --------------------------------------------------------------------------
# Sentiment engine
# --------------------------------------------------------------------------
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 64
FINBERT_MAX_LEN = 96

# --------------------------------------------------------------------------
# Forecasting model
# --------------------------------------------------------------------------
@dataclass
class LSTMConfig:
    lookback: int = 60           # trading days fed into the encoder
    horizon: int = 7             # 7-day trajectory, as per the proposal
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 120
    patience: int = 15           # early-stopping patience on validation loss
    seed: int = 42
    # Chronological, non-shuffled splits. No future data ever reaches training.
    train_end: str = "2022-06-30"
    val_end: str = "2023-03-31"
    feature_columns: list[str] = field(default_factory=lambda: [
        "log_return", "log_return_5d", "volatility_20d", "rsi_14",
        "macd_hist", "volume_z", "range_pct", "dist_ma50",
        "sent_mean", "sent_std", "sent_pos_ratio", "sent_neg_ratio",
        "headline_count_z", "sent_mean_5d", "sent_momentum",
    ])


LSTM = LSTMConfig()

# --------------------------------------------------------------------------
# Hybrid RAG
# --------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = RAG_INDEX / "chroma"
CHROMA_NEWS_COLLECTION = "sp500_headlines"
CHROMA_KB_COLLECTION = "investment_principles"
BM25_PATH = RAG_INDEX / "bm25.pkl"
KG_PATH = RAG_INDEX / "knowledge_graph.gpickle"

RRF_K = 60                # reciprocal-rank-fusion damping constant
RETRIEVAL_TOP_K = 8
CANDIDATE_K = 40          # per-retriever candidate depth before fusion
KG_EXPANSION_TERMS = 4    # neighbour entities injected into the expanded query

# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------
ANTHROPIC_MODEL = "claude-sonnet-5"
LLM_MAX_TOKENS = 1600


def anthropic_key() -> str | None:
    """Return an Anthropic key if the user configured one, else ``None``.

    Without a key the agent layer transparently falls back to a deterministic
    rule-based reasoner so the full pipeline still runs offline.
    """
    return os.environ.get("ANTHROPIC_API_KEY") or None


# --------------------------------------------------------------------------
# Decision thresholds
# --------------------------------------------------------------------------
# Calibrated on the VALIDATION period only, by
# ``finagent_pulse.evaluation.calibrate_decision_thresholds()``.
#
# A 7-day index forecast is inherently small relative to 7-day volatility: the
# signal-to-noise ratio on validation spans 0.005-0.50 and never approaches 1.
# Demanding a forecast larger than the noise band would therefore mean never
# trading at all. Instead both gates are set at the 80th percentile of the
# validation distribution, so the committee acts only on its most confident
# quintile of days and abstains on the rest.
def _calibrated(key: str, default: float) -> float:
    """Read a calibrated constant from the report that produced it.

    ``evaluation.calibrate_decision_thresholds()`` writes decision_thresholds.json.
    Reading it back here keeps the constant and its derivation from drifting apart;
    the literal is the fallback for a fresh checkout that has not run the pipeline.
    """
    try:
        import json
        return float(json.loads(
            (REPORTS / "decision_thresholds.json").read_text())[key])
    except Exception:
        return default


SIGNAL_TO_NOISE_MIN = _calibrated("signal_to_noise_min", 0.175)
# A floor for numerical triviality only -- NOT a second conviction gate.
# Gating on absolute return as well as on signal-to-noise would test the same
# quantity twice and, at any meaningful level, suppress every trade.
MIN_MATERIAL_RETURN = 0.001        # 0.1% over 7 sessions
SENTIMENT_STRONG = 0.25
HIGH_VOL_PERCENTILE = 0.80

# Fusion weights, calibrated by finagent_pulse.rag.evaluate.calibrate() on a
# development split of the benchmark (seed 7) that is disjoint from the split
# used to report the ablation results (seed 42).
# The dense arm is fixed at 1.0 and the others are expressed relative to it.
#
# Re-run after the BM25 tokeniser fix (rag/index.py::tokenize). The previous
# grid was searched against an index that kept punctuation glued to its terms,
# which understated the sparse arm and pulled its optimal weight down to 0.30.
#
# The 35-point dev grid spans nDCG@10 0.1503-0.2078, and the two axes behave
# very differently:
#   The BM25 weight matters, and matters more than it used to. The grid mean
#   climbs monotonically with it -- 0.1624, 0.1739, 0.1830, 0.1865, 0.1948,
#   0.1999, 0.2001 across 0.00 -> 1.00 -- so removing the sparse arm costs
#   0.038 nDCG and this weight is worth calibrating.
#   The KG weight does not. The grid mean falls monotonically as it rises
#   (0.1962 -> 0.1902 -> 0.1851 -> 0.1810 -> 0.1766) and half of the top six
#   settings have the expanded arms switched off entirely.
#
# The grid argmax is bm25 = 1.00, kg = 0.00 (0.2078). The configuration below,
# 0.80 / 0.25, ranks second at 0.2036, and a paired bootstrap over the per-query
# dev scores says the gap is not a difference: +0.0042, 95% CI [-0.008, +0.017],
# p = 0.51. Switching the KG arms off at the shipped sparse weight is likewise a
# non-difference (-0.0021, p = 0.74). The weights are therefore not determined
# by this data; 0.80 / 0.25 is kept because the argmax is not measurably better,
# not because the calibration demands it.
# See reports/TECHNICAL_REPORT.md section 5.1.
BM25_WEIGHT = 0.80
KG_ARM_WEIGHT = 0.25
