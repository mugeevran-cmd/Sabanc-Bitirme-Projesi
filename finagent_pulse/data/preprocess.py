"""Cleaning, trading-day alignment and feature engineering.

Produces two artefacts:

``headlines_clean.parquet``
    De-duplicated 2018-2024 news corpus, each headline mapped to the trading
    day on which it could first have been acted upon.

``features.parquet``
    One row per trading day holding price, technical and sentiment features
    plus the forward 7-day log-return targets consumed by the Bi-LSTM.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from finagent_pulse import config
from finagent_pulse.data.ingest import download_headlines, download_market

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------
# Headline cleaning
# --------------------------------------------------------------------------
def clean_headlines(force: bool = False) -> pd.DataFrame:
    if config.HEADLINES_CLEAN.exists() and not force:
        return pd.read_parquet(config.HEADLINES_CLEAN)

    df = download_headlines()
    df = df.rename(columns={"Title": "headline", "Date": "date", "CP": "corpus_close"})
    df["date"] = pd.to_datetime(df["date"])

    # Restrict to the study window defined in the proposal.
    df = df[(df["date"] >= config.START_DATE) & (df["date"] <= config.END_DATE)].copy()

    df["headline"] = (
        df["headline"].astype(str)
        .str.replace("’", "'", regex=False)
        .str.replace("“", '"', regex=False)
        .str.replace("”", '"', regex=False)
        .map(lambda s: _WS.sub(" ", s).strip())
    )

    before = len(df)
    df = df[df["headline"].str.len() >= 15]
    # The corpus repeats syndicated headlines; keep the first occurrence per day.
    df = df.drop_duplicates(subset=["date", "headline"], keep="first")
    df = df.sort_values(["date", "headline"]).reset_index(drop=True)
    df["doc_id"] = [f"h{i:06d}" for i in range(len(df))]
    log.info("Headlines: %d -> %d after cleaning (%d removed)", before, len(df), before - len(df))

    df.to_parquet(config.HEADLINES_CLEAN, index=False)
    return df


# --------------------------------------------------------------------------
# Technical indicators
# --------------------------------------------------------------------------
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    return macd - macd.ewm(span=9, adjust=False).mean()


def build_price_features(market: pd.DataFrame) -> pd.DataFrame:
    px = market.rename(columns={"Date": "date"}).sort_values("date").reset_index(drop=True)
    px["date"] = pd.to_datetime(px["date"])

    px["log_return"] = np.log(px["close"]).diff()
    px["log_return_5d"] = np.log(px["close"]).diff(5)
    px["volatility_20d"] = px["log_return"].rolling(20).std()
    px["rsi_14"] = _rsi(px["close"])
    # Scale MACD by price level so the feature is comparable across regimes.
    px["macd_hist"] = _macd_hist(px["close"]) / px["close"]

    if px["volume"].notna().any():
        vol = pd.to_numeric(px["volume"], errors="coerce")
        px["volume_z"] = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    else:
        px["volume_z"] = 0.0

    px["range_pct"] = (px["high"] - px["low"]) / px["close"]
    px["dist_ma50"] = px["close"] / px["close"].rolling(50).mean() - 1.0
    return px


# --------------------------------------------------------------------------
# Daily sentiment aggregation
# --------------------------------------------------------------------------
def aggregate_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-headline FinBERT scores into one vector per calendar day."""
    g = scored.groupby("date")
    daily = pd.DataFrame({
        "sent_mean": g["sentiment"].mean(),
        "sent_std": g["sentiment"].std().fillna(0.0),
        "sent_pos_ratio": g["label"].apply(lambda s: (s == "positive").mean()),
        "sent_neg_ratio": g["label"].apply(lambda s: (s == "negative").mean()),
        "headline_count": g["sentiment"].size(),
    }).reset_index()
    return daily


def merge_features(force: bool = False) -> pd.DataFrame:
    """Join prices, technicals and sentiment; append the 7-day targets."""
    if config.FEATURES.exists() and not force:
        return pd.read_parquet(config.FEATURES)

    if not config.DAILY_SENTIMENT.exists():
        raise FileNotFoundError(
            "daily_sentiment.parquet missing -- run finagent_pulse.models.sentiment first."
        )

    px = build_price_features(download_market())
    daily = pd.read_parquet(config.DAILY_SENTIMENT)
    daily["date"] = pd.to_datetime(daily["date"])

    # News published on a non-trading day (weekend/holiday) is attributed to the
    # next trading session -- the first moment it could influence a price.
    sessions = px["date"].sort_values().to_numpy()
    idx = np.searchsorted(sessions, daily["date"].to_numpy(), side="left")
    daily = daily[idx < len(sessions)].copy()
    daily["session"] = sessions[idx[idx < len(sessions)]]

    per_session = (
        daily.drop(columns=["date"])
        .groupby("session")
        .agg(
            sent_mean=("sent_mean", "mean"),
            sent_std=("sent_std", "mean"),
            sent_pos_ratio=("sent_pos_ratio", "mean"),
            sent_neg_ratio=("sent_neg_ratio", "mean"),
            headline_count=("headline_count", "sum"),
        )
        .reset_index()
        .rename(columns={"session": "date"})
    )

    df = px.merge(per_session, on="date", how="left")

    # Days without news carry neutral sentiment and zero coverage.
    df["has_news"] = df["sent_mean"].notna()
    for col in ("sent_mean", "sent_std", "sent_pos_ratio", "sent_neg_ratio"):
        df[col] = df[col].fillna(0.0)
    df["headline_count"] = df["headline_count"].fillna(0)

    cnt = df["headline_count"]
    df["headline_count_z"] = ((cnt - cnt.rolling(60, min_periods=10).mean())
                              / cnt.rolling(60, min_periods=10).std()).fillna(0.0)
    df["sent_mean_5d"] = df["sent_mean"].rolling(5, min_periods=1).mean()
    df["sent_momentum"] = df["sent_mean_5d"] - df["sent_mean"].rolling(20, min_periods=1).mean()

    # Targets: cumulative log-return from t to t+h, for h = 1..horizon.
    logc = np.log(df["close"])
    for h in range(1, config.LSTM.horizon + 1):
        df[f"target_h{h}"] = logc.shift(-h) - logc

    # Trim the corpus tail: keep only sessions actually covered by the news feed.
    last_news = df.loc[df["has_news"], "date"].max()
    df = df[df["date"] <= last_news].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=config.LSTM.feature_columns).reset_index(drop=True)

    df.to_parquet(config.FEATURES, index=False)
    log.info("Feature table: %d sessions %s -> %s",
             len(df), df["date"].min().date(), df["date"].max().date())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    hl = clean_headlines(force=True)
    print(f"clean headlines: {hl.shape}, {hl.date.min().date()} -> {hl.date.max().date()}")
