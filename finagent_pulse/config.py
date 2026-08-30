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
    # Upper bound on the test window. None means "everything after val_end",
    # which is what the shipped split does; walk-forward folds set it so each
    # fold is scored on its own regime rather than on everything that follows.
    test_end: str | None = None
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
# The 35-point dev grid spans nDCG@10 0.1389-0.1756, and the two axes behave
# very differently:
#   The BM25 weight matters. At w=0 the grid mean is 0.1472, against 0.1727
#   across 0.45-0.80, so the sparse arm carries real signal and this weight is
#   worth calibrating.
#   The KG weight barely does. The best configuration with the expanded arms
#   switched off entirely scores 0.1736 against 0.1756 with them on (+1.1%), and
#   the grid mean falls monotonically above 0.25 (0.1698 -> 0.1667 -> 0.1643 ->
#   0.1619). 0.25 is the argmax of a genuinely shallow optimum; the arm is kept
#   for what it does on keyword queries, not because the calibration demands it.
#   See reports/TECHNICAL_REPORT.md section 5.4.
BM25_WEIGHT = 0.80
KG_ARM_WEIGHT = 0.25
