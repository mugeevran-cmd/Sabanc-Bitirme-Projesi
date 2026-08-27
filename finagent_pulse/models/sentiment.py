"""FinBERT sentiment engine.

Scores every headline with ProsusAI/finbert and reduces the three-way softmax
to a single signed intensity in [-1, 1]:

    sentiment = P(positive) - P(negative)

so that a confidently bullish headline sits near +1, a confidently bearish one
near -1, and a neutral or ambiguous one near 0.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from finagent_pulse import config
from finagent_pulse.data.preprocess import aggregate_sentiment, clean_headlines

log = logging.getLogger(__name__)

_LABELS = ("positive", "negative", "neutral")   # ProsusAI/finbert label order


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FinBertScorer:
    """Thin batched wrapper around the FinBERT sequence classifier."""

    def __init__(self, model_name: str = config.FINBERT_MODEL) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        # Trust the checkpoint's own id2label mapping rather than assuming order.
        id2label = getattr(self.model.config, "id2label", None) or {}
        self.labels = [str(id2label.get(i, _LABELS[i])).lower() for i in range(3)]
        self.pos_ix = self.labels.index("positive")
        self.neg_ix = self.labels.index("negative")
        log.info("FinBERT loaded on %s (labels=%s)", self.device, self.labels)

    @torch.no_grad()
    def score(self, texts: list[str], batch_size: int = config.FINBERT_BATCH_SIZE) -> pd.DataFrame:
        probs_all: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=config.FINBERT_MAX_LEN, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs_all.append(torch.softmax(logits, dim=-1).cpu().numpy())
            if start and start % (batch_size * 20) == 0:
                log.info("  scored %d/%d headlines", start, len(texts))

        probs = np.vstack(probs_all)
        out = pd.DataFrame({
            f"p_{lab}": probs[:, i] for i, lab in enumerate(self.labels)
        })
        out["sentiment"] = probs[:, self.pos_ix] - probs[:, self.neg_ix]
        out["confidence"] = probs.max(axis=1)
        out["label"] = [self.labels[i] for i in probs.argmax(axis=1)]
        return out


def score_corpus(force: bool = False) -> pd.DataFrame:
    """Score every cleaned headline and cache the per-headline result."""
    if config.HEADLINES_SCORED.exists() and not force:
        return pd.read_parquet(config.HEADLINES_SCORED)

    hl = clean_headlines()
    scorer = FinBertScorer()
    log.info("Scoring %d headlines...", len(hl))
    scores = scorer.score(hl["headline"].tolist())

    out = pd.concat([hl.reset_index(drop=True), scores], axis=1)
    out.to_parquet(config.HEADLINES_SCORED, index=False)

    daily = aggregate_sentiment(out)
    daily.to_parquet(config.DAILY_SENTIMENT, index=False)
    log.info("Wrote %d scored headlines / %d daily sentiment rows", len(out), len(daily))
    return out


# --------------------------------------------------------------------------
# Fear & Greed index
# --------------------------------------------------------------------------
def fear_greed_index(features: pd.DataFrame, window: int = 20) -> pd.Series:
    """Map smoothed daily sentiment onto a 0-100 'Market Fear & Greed' scale.

    The raw signed sentiment is smoothed over ``window`` sessions and then
    percentile-ranked against the full study period, so 50 means "as calm as a
    typical day in 2018-2024" rather than an arbitrary absolute level.
    """
    smoothed = features["sent_mean"].rolling(window, min_periods=5).mean()
    return (smoothed.rank(pct=True) * 100).round(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scored = score_corpus(force=True)
    print(scored[["date", "headline", "sentiment", "label"]].head(8).to_string(index=False))
    print("\nlabel distribution:")
    print(scored["label"].value_counts(normalize=True).round(4))
    print(f"\nmean sentiment: {scored['sentiment'].mean():.4f}")
