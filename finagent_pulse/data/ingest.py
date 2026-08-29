"""Data-layer ingestion: Kaggle news corpus + yfinance market data."""
from __future__ import annotations

import io
import logging
import urllib.request
import zipfile

import pandas as pd

from finagent_pulse import config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. News corpus
# --------------------------------------------------------------------------
def download_headlines(force: bool = False) -> pd.DataFrame:
    """Fetch the Kaggle S&P 500 headline corpus (2008-2024) and cache it."""
    if config.HEADLINES_CSV.exists() and not force:
        log.info("Using cached headline corpus at %s", config.HEADLINES_CSV)
        return pd.read_csv(config.HEADLINES_CSV)

    log.info("Downloading %s", config.KAGGLE_DATASET)
    try:
        req = urllib.request.Request(
            config.KAGGLE_DOWNLOAD_URL, headers={"User-Agent": "finagent-pulse"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = resp.read()
    except Exception as exc:
        # Every downstream stage needs this file, so fail with instructions
        # rather than an opaque URLError from four frames down.
        raise RuntimeError(
            f"Could not download the Kaggle corpus ({exc}).\n"
            f"Kaggle rate-limits and sometimes requires a signed-in session. "
            f"Download it manually from\n"
            f"  https://www.kaggle.com/datasets/{config.KAGGLE_DATASET}\n"
            f"and place the CSV at {config.HEADLINES_CSV}, then re-run."
        ) from exc

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        zf.extract(csv_name, config.DATA_RAW)

    extracted = config.DATA_RAW / csv_name
    if extracted != config.HEADLINES_CSV:
        extracted.rename(config.HEADLINES_CSV)
    return pd.read_csv(config.HEADLINES_CSV)


# --------------------------------------------------------------------------
# 2. Market data
# --------------------------------------------------------------------------
def _market_from_corpus() -> pd.DataFrame:
    """Reconstruct a close-only price series from the corpus ``CP`` column.

    Used when yfinance is unreachable (offline runs, rate limiting). The corpus
    stores one closing price per trading day, so it is a faithful -- if
    OHLCV-less -- substitute for the index close.
    """
    raw = download_headlines()
    px = (
        raw.assign(Date=pd.to_datetime(raw["Date"]))
        .groupby("Date", as_index=False)["CP"]
        .first()
        .rename(columns={"CP": "close"})
        .sort_values("Date")
    )
    px["open"] = px["close"]
    px["high"] = px["close"]
    px["low"] = px["close"]
    px["volume"] = pd.NA
    px["source"] = "kaggle_cp"
    log.warning(
        "Using corpus close prices: no OHLC and no volume. 'range_pct' and "
        "'volume_z' will be constant 0 and carry no information -- 2 of the 15 "
        "model features are dead in this mode. Results are NOT comparable with "
        "a yfinance run; re-run with network access before reporting metrics.")
    return px


def download_market(force: bool = False) -> pd.DataFrame:
    """Download S&P 500 OHLCV from yfinance, with a corpus-derived fallback."""
    if config.MARKET_CSV.exists() and not force:
        df = pd.read_csv(config.MARKET_CSV, parse_dates=["Date"])
        log.info("Using cached market data (%d rows)", len(df))
        return df

    frame: pd.DataFrame | None = None
    try:
        import yfinance as yf

        raw = yf.download(
            config.TICKER,
            start=config.START_DATE,
            end=config.END_DATE,
            auto_adjust=False,
            progress=False,
        )
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):        # yfinance >= 0.2.51
                raw.columns = raw.columns.get_level_values(0)
            frame = (
                raw.reset_index()
                .rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                })
                [["Date", "open", "high", "low", "close", "volume"]]
            )
            frame["source"] = "yfinance"
            log.info("yfinance returned %d trading days", len(frame))
    except Exception as exc:                                   # pragma: no cover
        log.warning("yfinance unavailable (%s); falling back to corpus prices", exc)

    if frame is None or frame.empty:
        # Note this fallback also needs the corpus download, so a network
        # failure that killed yfinance may well kill this too -- the error from
        # download_headlines() says what to do about it.
        frame = _market_from_corpus()

    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame[
        (frame["Date"] >= config.START_DATE) & (frame["Date"] <= config.END_DATE)
    ].sort_values("Date").reset_index(drop=True)
    frame.to_csv(config.MARKET_CSV, index=False)
    return frame


def ingest_all(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(headlines, market)`` raw frames."""
    return download_headlines(force=force), download_market(force=force)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    news, market = ingest_all()
    print(f"headlines : {news.shape}  {news['Date'].min()} -> {news['Date'].max()}")
    print(f"market    : {market.shape}  source={market['source'].iloc[0]}")
