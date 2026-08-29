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
#
# One definition, exposed two ways: ``fear_greed_index`` for a whole series and
# ``fear_greed_at`` for a single decision date. Both smooth first and rank
# point-in-time, so the dashboard and the committee can never disagree about
# what the number is.
# --------------------------------------------------------------------------
FEAR_GREED_WINDOW = 20         # sessions of smoothing before ranking
FEAR_GREED_MIN_HISTORY = 20    # ranked sessions needed before a value means anything


def _smoothed_sentiment(features: pd.DataFrame, window: int) -> pd.Series:
    return features["sent_mean"].rolling(window, min_periods=5).mean()


def fear_greed_index(features: pd.DataFrame,
                     window: int = FEAR_GREED_WINDOW) -> pd.Series:
    """Map smoothed daily sentiment onto a 0-100 'Market Fear & Greed' scale.

    The raw signed sentiment is smoothed over ``window`` sessions and then
    percentile-ranked *point-in-time*: each session is ranked only against the
    sessions that preceded it. 50 therefore means "as calm as a typical day so
    far", and the value a historical date carries is one that could actually
    have been computed on that date.

    Ranking against the whole 2018-2024 sample instead would leak the future
    into every backtested decision, so the expanding form is the only one
    exposed.
    """
    smoothed = _smoothed_sentiment(features, window).dropna()
    ranked = (smoothed.expanding(min_periods=FEAR_GREED_MIN_HISTORY)
              .apply(lambda w: (w <= w[-1]).mean(), raw=True) * 100)
    return ranked.reindex(features.index).round(1)


def fear_greed_at(features: pd.DataFrame,
                  window: int = FEAR_GREED_WINDOW) -> float:
    """Fear & Greed for the *last* row of ``features``.

    ``features`` must already be truncated to the decision date; the percentile
    is taken against that history alone. Returns a neutral 50.0 while there is
    not yet enough history to rank against.

    Equivalent by construction to ``fear_greed_index(features).iloc[-1]``, but
    without recomputing the whole expanding series for one date.
    """
    smoothed = _smoothed_sentiment(features, window).dropna()
    if len(smoothed) < FEAR_GREED_MIN_HISTORY:
        return 50.0
    return round(float((smoothed <= smoothed.iloc[-1]).mean() * 100), 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scored = score_corpus(force=True)
    print(scored[["date", "headline", "sentiment", "label"]].head(8).to_string(index=False))
    print("\nlabel distribution:")
    print(scored["label"].value_counts(normalize=True).round(4))
    print(f"\nmean sentiment: {scored['sentiment'].mean():.4f}")
