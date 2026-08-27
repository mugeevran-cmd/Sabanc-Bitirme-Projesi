"""Bi-directional LSTM forecaster for a 7-day S&P 500 return trajectory.

Design notes
------------
*Return space, not price space.*  The network predicts cumulative log-returns
``log(P_{t+h}/P_t)`` for h = 1..7 rather than raw price levels.  Regressing on
levels yields a near-perfect R^2 that merely reflects the fact that tomorrow's
price is close to today's; regressing on returns is the honest test.

*Leak-safe splits.*  Windows are assigned to train/val/test by their forecast
origin, and any window whose 7-day target horizon would reach into the next
split is purged (an embargo).  The feature scaler is fitted on training rows
only.

*Bidirectionality.*  The encoder reads the 60-day lookback window in both
directions.  The whole window lies strictly in the past relative to the
forecast origin, so no future information leaks.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score

from finagent_pulse import config
from finagent_pulse.data.preprocess import merge_features

log = logging.getLogger(__name__)

CKPT = config.MODELS_OUT / "bilstm.pt"
SCALER_PATH = config.MODELS_OUT / "scaler.npz"
METRICS_PATH = config.MODELS_OUT / "forecast_metrics.json"
PREDICTIONS_PATH = config.MODELS_OUT / "test_predictions.parquet"


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class BiLSTMForecaster(nn.Module):
    def __init__(self, n_features: int, cfg: config.LSTMConfig = config.LSTM) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        enc_dim = cfg.hidden_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(enc_dim, cfg.hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, cfg.horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                 # (B, T, 2H)
        # Concatenate the final step of the forward pass with the first step of
        # the backward pass -- the two ends of the bidirectional encoding.
        h = out.shape[-1] // 2
        enc = torch.cat([out[:, -1, :h], out[:, 0, h:]], dim=-1)
        return self.head(enc)


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------
def build_windows(df: pd.DataFrame, feature_cols: list[str], cfg: config.LSTMConfig):
    """Return sliding windows plus the forecast-origin date of each window."""
    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df[[f"target_h{h}" for h in range(1, cfg.horizon + 1)]].to_numpy(dtype=np.float32)
    dates = df["date"].to_numpy()
    closes = df["close"].to_numpy(dtype=np.float64)

    xs, ys, origins, origin_close = [], [], [], []
    last_origin = len(df) - cfg.horizon          # need h future closes for the target
    for end in range(cfg.lookback, last_origin):
        xs.append(X_raw[end - cfg.lookback:end])
        ys.append(y_raw[end - 1])                # target anchored at the origin bar
        origins.append(dates[end - 1])
        origin_close.append(closes[end - 1])

    return (np.asarray(xs), np.asarray(ys),
            pd.to_datetime(pd.Series(origins)), np.asarray(origin_close))


def split_windows(origins: pd.Series, cfg: config.LSTMConfig):
    """Chronological train/val/test masks with an embargo at each boundary.

    A window is dropped when its forecast horizon would cross the boundary,
    so no training target overlaps a validation or test observation.
    """
    train_end = pd.Timestamp(cfg.train_end)
    val_end = pd.Timestamp(cfg.val_end)
    embargo = pd.Timedelta(days=cfg.horizon * 2)   # calendar slack for weekends

    train = origins <= (train_end - embargo)
    val = (origins > train_end) & (origins <= val_end - embargo)
    test = origins > val_end
    return train.to_numpy(), val.to_numpy(), test.to_numpy()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, origin_close: np.ndarray) -> dict:
    """Return-space and price-space metrics, plus directional accuracy."""
    h = y_true.shape[1]
    per_h = []
    for i in range(h):
        per_h.append({
            "horizon": i + 1,
            "rmse_return": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "r2_return": float(r2_score(y_true[:, i], y_pred[:, i])),
            "directional_accuracy": float(np.mean(
                np.sign(y_pred[:, i]) == np.sign(y_true[:, i]))),
        })

    # Reconstructed price levels: P_t * exp(cumulative log-return).
    price_true = origin_close[:, None] * np.exp(y_true)
    price_pred = origin_close[:, None] * np.exp(y_pred)

    # Persistence baseline: "the next 7 days return exactly zero".
    naive_rmse = float(np.sqrt(mean_squared_error(y_true, np.zeros_like(y_true))))
    model_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {
        "n_samples": int(len(y_true)),
        "rmse_return_overall": model_rmse,
        "r2_return_overall": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "rmse_price_overall": float(np.sqrt(mean_squared_error(price_true, price_pred))),
        "r2_price_overall": float(r2_score(price_true, price_pred, multioutput="variance_weighted")),
        "mape_price_pct": float(np.mean(np.abs(price_pred - price_true) / price_true) * 100),
        "directional_accuracy_h7": per_h[-1]["directional_accuracy"],
        "naive_zero_rmse_return": naive_rmse,
        "skill_vs_naive_pct": float((1 - model_rmse / naive_rmse) * 100),
        "per_horizon": per_h,
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(df: pd.DataFrame | None = None,
          feature_cols: list[str] | None = None,
          cfg: config.LSTMConfig = config.LSTM,
          save: bool = True,
          tag: str = "full",
          seeds: tuple[int, ...] = (42, 1337, 2024)) -> dict:
    """Train a seed-ensembled Bi-LSTM and report validation / test metrics.

    Several independently seeded members are averaged: a single LSTM on ~1k
    windows of noisy financial data has high initialisation variance, and the
    ensemble mean is a materially more stable estimator.
    """
    df = merge_features() if df is None else df
    feature_cols = feature_cols or cfg.feature_columns
    device = _device()

    X, y, origins, origin_close = build_windows(df, feature_cols, cfg)
    m_tr, m_va, m_te = split_windows(origins, cfg)
    log.info("[%s] windows train=%d val=%d test=%d (%d features)",
             tag, m_tr.sum(), m_va.sum(), m_te.sum(), len(feature_cols))

    # Scaler fitted on training windows only -- no test statistics leak in.
    flat = X[m_tr].reshape(-1, X.shape[-1])
    mu, sigma = flat.mean(axis=0), flat.std(axis=0) + 1e-8
    Xs = (X - mu) / sigma

    # Targets rescaled per horizon so the 1-day and 7-day heads contribute
    # comparably to the loss instead of the long horizon dominating it.
    y_sigma = y[m_tr].std(axis=0) + 1e-8
    ys = y / y_sigma

    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    Xtr, ytr = to_t(Xs[m_tr]), to_t(ys[m_tr])
    Xva, yva = to_t(Xs[m_va]), to_t(ys[m_va])
    Xte = to_t(Xs[m_te])

    members, val_preds, test_preds, val_losses = [], [], [], []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = BiLSTMForecaster(len(feature_cols), cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
        # Huber on standardised targets: robust to the fat tails of equity returns.
        loss_fn = nn.HuberLoss(delta=1.0)

        best_val, best_state, bad_epochs = float("inf"), None, 0
        n = len(Xtr)

        for epoch in range(cfg.epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xtr[idx]), ytr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xva), yva).item()
            sched.step(val_loss)

            if val_loss < best_val - 1e-7:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_preds.append(model(Xva).cpu().numpy() * y_sigma)
            test_preds.append(model(Xte).cpu().numpy() * y_sigma)
        members.append({k: v.cpu() for k, v in best_state.items()})
        val_losses.append(best_val)
        log.info("[%s] seed %d done (val loss %.6f, %d epochs)", tag, seed, best_val, epoch + 1)

    pred_va = np.mean(val_preds, axis=0)
    pred_te = np.mean(test_preds, axis=0)

    metrics = {
        "tag": tag,
        "features": feature_cols,
        "config": asdict(cfg),
        "seeds": list(seeds),
        "member_val_losses": val_losses,
        "validation": evaluate(y[m_va], pred_va, origin_close[m_va]),
        "test": evaluate(y[m_te], pred_te, origin_close[m_te]),
        "split_dates": {
            "train_range": [str(origins[m_tr].min().date()), str(origins[m_tr].max().date())],
            "val_range": [str(origins[m_va].min().date()), str(origins[m_va].max().date())],
            "test_range": [str(origins[m_te].min().date()), str(origins[m_te].max().date())],
        },
    }

    if save:
        torch.save({"members": members,
                    "features": feature_cols,
                    "config": asdict(cfg),
                    "y_sigma": y_sigma}, CKPT)
        np.savez(SCALER_PATH, mu=mu, sigma=sigma, y_sigma=y_sigma)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        pd.DataFrame({
            "date": origins[m_te].to_numpy(),
            "close": origin_close[m_te],
            **{f"true_h{i+1}": y[m_te][:, i] for i in range(cfg.horizon)},
            **{f"pred_h{i+1}": pred_te[:, i] for i in range(cfg.horizon)},
        }).to_parquet(PREDICTIONS_PATH, index=False)

    return metrics


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
class ForecastService:
    """Loads the trained checkpoint and produces a 7-day trajectory on demand."""

    def __init__(self) -> None:
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        self.features: list[str] = ckpt["features"]
        self.cfg = config.LSTMConfig(**{
            k: v for k, v in ckpt["config"].items()
            if k in config.LSTMConfig.__dataclass_fields__
        })
        self.device = _device()
        self.y_sigma = np.asarray(ckpt["y_sigma"])

        self.models = []
        for state in ckpt["members"]:
            m = BiLSTMForecaster(len(self.features), self.cfg).to(self.device)
            m.load_state_dict(state)
            m.eval()
            self.models.append(m)

        stats = np.load(SCALER_PATH)
        self.mu, self.sigma = stats["mu"], stats["sigma"]

    @torch.no_grad()
    def forecast(self, df: pd.DataFrame, as_of: str | pd.Timestamp | None = None) -> dict:
        """Forecast the 7 sessions following ``as_of`` (default: last row)."""
        df = df.sort_values("date").reset_index(drop=True)
        if as_of is not None:
            df = df[df["date"] <= pd.Timestamp(as_of)]
        if len(df) < self.cfg.lookback:
            raise ValueError(f"need >= {self.cfg.lookback} sessions, got {len(df)}")

        window = df[self.features].to_numpy(dtype=np.float32)[-self.cfg.lookback:]
        x = torch.tensor(((window - self.mu) / self.sigma)[None, ...],
                         dtype=torch.float32, device=self.device)
        # Ensemble mean, de-standardised back into log-return space.
        cum_log = np.mean([m(x).cpu().numpy()[0] for m in self.models], axis=0) * self.y_sigma

        origin = df.iloc[-1]
        prices = float(origin["close"]) * np.exp(cum_log)
        return {
            "as_of": pd.Timestamp(origin["date"]),
            "origin_close": float(origin["close"]),
            "cumulative_log_returns": cum_log.tolist(),
            "prices": prices.tolist(),
            "total_return_pct": float((np.exp(cum_log[-1]) - 1) * 100),
            "path_return_pct": [(float(np.exp(c) - 1) * 100) for c in cum_log],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    m = train()
    t = m["test"]
    print(json.dumps({k: v for k, v in t.items() if k != "per_horizon"}, indent=2))
    print("\nper-horizon (test):")
    for row in t["per_horizon"]:
        print(f"  h={row['horizon']}  RMSE={row['rmse_return']:.5f}  "
              f"R2={row['r2_return']:+.4f}  DA={row['directional_accuracy']:.3f}")
