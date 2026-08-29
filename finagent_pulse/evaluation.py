"""Whole-system evaluation: forecaster ablation, sentiment validation, backtest.

Everything here reports on the held-out test period only (after 2023-03-31),
which no model, scaler or fusion weight ever saw during fitting.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy import stats

from finagent_pulse import config
from finagent_pulse.data.preprocess import merge_features
from finagent_pulse.models import forecaster

log = logging.getLogger(__name__)

FEATURE_ABLATION_PATH = config.REPORTS / "forecaster_ablation.csv"
SENTIMENT_EVAL_PATH = config.REPORTS / "sentiment_validation.json"
BACKTEST_PATH = config.REPORTS / "committee_backtest.csv"
SUMMARY_PATH = config.REPORTS / "evaluation_summary.json"


# --------------------------------------------------------------------------
# 1. Does the sentiment channel actually help the forecaster?
# --------------------------------------------------------------------------
def forecaster_ablation(save: bool = True) -> pd.DataFrame:
    df = merge_features()
    price_only = [c for c in config.LSTM.feature_columns if not c.startswith(
        ("sent_", "headline_"))]
    sent_only = [c for c in config.LSTM.feature_columns if c.startswith(
        ("sent_", "headline_"))]

    variants = {
        "price_only": price_only,
        "sentiment_only": sent_only,
        "price_plus_sentiment": config.LSTM.feature_columns,
    }

    rows = []
    for name, cols in variants.items():
        m = forecaster.train(df, feature_cols=cols, save=False, tag=name)
        t = m["test"]
        rows.append({
            "variant": name,
            "n_features": len(cols),
            "rmse_return": t["rmse_return_overall"],
            "r2_return": t["r2_return_overall"],
            "r2_price": t["r2_price_overall"],
            "mape_price_pct": t["mape_price_pct"],
            "directional_acc_h7": t["directional_accuracy_h7"],
            "skill_vs_naive_pct": t["skill_vs_naive_pct"],
        })
        log.info("%-22s RMSE=%.5f  DA(h7)=%.3f  skill=%+.2f%%",
                 name, t["rmse_return_overall"], t["directional_accuracy_h7"],
                 t["skill_vs_naive_pct"])

    out = pd.DataFrame(rows)
    if save:
        out.to_csv(FEATURE_ABLATION_PATH, index=False)
    return out


# --------------------------------------------------------------------------
# 2. Is the sentiment signal economically meaningful?
# --------------------------------------------------------------------------
def _block_bootstrap_p(x, y, n_boot: int = 5000, block: int = 10,
                       seed: int = 42) -> tuple[float, float]:
    """Correlation and a two-sided p-value that survives autocorrelation.

    ``scipy.stats.pearsonr`` assumes i.i.d. observations. Daily sentiment and
    daily returns are both serially correlated, so its p-value is far smaller
    than the evidence warrants -- with 230 sessions it reports 1e-21 for a
    relationship a dependence-aware test puts at 2e-4.

    The null here is built by resampling ``y`` in contiguous circular blocks:
    that destroys any association with ``x`` while preserving ``y``'s own
    short-range persistence, so the reference distribution has the same
    autocorrelation as the real series. p is the share of bootstrap draws whose
    |r| reaches the observed |r|.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    r_obs = float(np.corrcoef(x, y)[0, 1])
    ext = np.concatenate([y, y[:block]])          # wrap around for circularity
    n_blocks = -(-n // block)
    hits = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        yb = np.concatenate([ext[s:s + block] for s in starts])[:n]
        if abs(float(np.corrcoef(x, yb)[0, 1])) >= abs(r_obs):
            hits += 1
    return r_obs, (hits + 1) / (n_boot + 1)



def sentiment_validation(save: bool = True) -> dict:
    """Validate FinBERT output against realised market behaviour.

    There are no human sentiment labels for this corpus, so the engine is
    validated the way it is actually used: against the market. Three checks --
    correlation with contemporaneous returns, next-day directional hit rate,
    and behaviour in the tails -- with the test period held out.
    """
    df = merge_features()
    test = df[df["date"] > pd.Timestamp(config.LSTM.val_end)].copy()

    test["next_return"] = np.log(test["close"]).diff().shift(-1)
    valid = test.dropna(subset=["next_return", "log_return", "sent_mean"])

    same_r, same_p = stats.pearsonr(valid["sent_mean"], valid["log_return"])
    next_r, next_p = stats.pearsonr(valid["sent_mean"], valid["next_return"])
    # Both series are autocorrelated, so the i.i.d. p-values above are reported
    # alongside a block-bootstrap p rather than on their own.
    _, same_p_boot = _block_bootstrap_p(valid["sent_mean"], valid["log_return"])
    _, next_p_boot = _block_bootstrap_p(valid["sent_mean"], valid["next_return"])

    signed = valid[valid["sent_mean"].abs() >= 0.05]
    hit_rate = float((np.sign(signed["sent_mean"]) ==
                      np.sign(signed["next_return"])).mean())

    q = valid["sent_mean"].quantile([0.2, 0.8])
    bearish = valid[valid["sent_mean"] <= q.iloc[0]]["next_return"]
    bullish = valid[valid["sent_mean"] >= q.iloc[1]]["next_return"]
    t_stat, t_p = stats.ttest_ind(bullish, bearish, equal_var=False)

    result = {
        "period": [str(valid["date"].min().date()), str(valid["date"].max().date())],
        "n_sessions": int(len(valid)),
        # "p_value" keeps its original meaning (the i.i.d. Pearson p) so older
        # report files and their readers stay valid; the dependence-aware p is
        # added beside it and is the one to quote.
        "corr_same_day": {"r": float(same_r), "p_value": float(same_p),
                          "p_value_block_bootstrap": float(same_p_boot)},
        "corr_next_day": {"r": float(next_r), "p_value": float(next_p),
                          "p_value_block_bootstrap": float(next_p_boot)},
        "next_day_hit_rate": hit_rate,
        "n_signed_sessions": int(len(signed)),
        "tail_spread": {
            "bullish_quintile_mean_next_return_bps": float(bullish.mean() * 10000),
            "bearish_quintile_mean_next_return_bps": float(bearish.mean() * 10000),
            "welch_t": float(t_stat),
            "p_value": float(t_p),
        },
        "label_distribution": pd.read_parquet(config.HEADLINES_SCORED)["label"]
            .value_counts(normalize=True).round(4).to_dict(),
    }
    log.info("sentiment: same-day r=%.3f (iid p=%.3g, block p=%.3g), "
             "next-day r=%.3f (iid p=%.3g, block p=%.3g), hit=%.3f",
             same_r, same_p, same_p_boot, next_r, next_p, next_p_boot, hit_rate)
    if save:
        SENTIMENT_EVAL_PATH.write_text(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------
# 2b. Decision-threshold calibration (validation period)
# --------------------------------------------------------------------------
THRESHOLD_PATH = config.REPORTS / "decision_thresholds.json"


def calibrate_decision_thresholds(percentile: float = 0.80,
                                  stride: int = 2,
                                  save: bool = True) -> dict:
    """Derive the committee's conviction gates from the VALIDATION period.

    The values this returns are the ones hard-coded in ``config`` as
    ``SIGNAL_TO_NOISE_MIN``. It is kept as a
    runnable function so the constants are reproducible rather than arbitrary.
    Only validation rows are used; the test period is untouched.
    """
    from finagent_pulse.models.forecaster import ForecastService

    df = merge_features()
    svc = ForecastService()
    val = df[(df["date"] > pd.Timestamp(config.LSTM.train_end)) &
             (df["date"] <= pd.Timestamp(config.LSTM.val_end))]

    rows = []
    for d in val["date"].iloc[::stride]:
        fc = svc.forecast(df, as_of=d)
        row = df[df["date"] == d].iloc[0]
        horizon_vol = float(row["volatility_20d"]) * np.sqrt(config.LSTM.horizon)
        magnitude = abs(fc["total_return_pct"]) / 100.0
        rows.append({"abs_forecast": magnitude,
                     "snr": magnitude / (horizon_vol + 1e-9)})

    v = pd.DataFrame(rows)
    result = {
        "percentile": percentile,
        "n_validation_points": int(len(v)),
        "signal_to_noise_min": float(v["snr"].quantile(percentile)),
        "return_threshold": float(v["abs_forecast"].quantile(percentile)),
        "snr_max_observed": float(v["snr"].max()),
        "note": ("signal-to-noise never approaches 1.0, so the gate is a "
                 "relative conviction percentile rather than an absolute "
                 "forecast-beats-noise test"),
    }
    log.info("calibrated: snr_min=%.3f return_threshold=%.4f (max snr seen %.3f)",
             result["signal_to_noise_min"], result["return_threshold"],
             result["snr_max_observed"])
    if save:
        THRESHOLD_PATH.write_text(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------
# 3. Committee backtest
# --------------------------------------------------------------------------
def committee_backtest(step: int = 3, save: bool = True) -> pd.DataFrame:
    """Run the full agent committee across the test period.

    Every ``step`` sessions the committee issues a directive, which is scored
    against the realised forward 7-day return. This measures the decision
    layer, not just the forecaster.
    """
    from finagent_pulse.agents.committee import run_committee

    df = merge_features()
    test = df[df["date"] > pd.Timestamp(config.LSTM.val_end)].reset_index(drop=True)
    # Leave the horizon at the end so every decision has a realised outcome.
    decision_rows = test.iloc[:-config.LSTM.horizon:step]
    log.info("Backtesting %d committee decisions", len(decision_rows))

    rows = []
    for i, row in enumerate(decision_rows.itertuples()):
        # Findings only: the prose would be discarded and, with an API key
        # set, would cost one model call per agent per decision.
        state = run_committee(df, row.date, narrative=False)
        risk, quant, sent = state["risk"], state["quant"], state["sentiment"]
        rows.append({
            "date": row.date,
            "close": row.close,
            "directive": risk["directive"],
            "position_pct": risk["position_pct"],
            "conviction": risk["conviction"],
            "agreement": risk["agreement"],
            "trap_flagged": risk["trap_warning"] is not None,
            "forecast_7d_pct": quant["forecast_7d_pct"],
            "signal_to_noise": quant["signal_to_noise"],
            "volatility_regime": quant["volatility_regime"],
            "sentiment": sent["sentiment_now"],
            "fear_greed": sent["fear_greed_index"],
            "realised_7d_pct": float(getattr(row, f"target_h{config.LSTM.horizon}") * 100),
        })
        if i % 10 == 0:
            log.info("  %d/%d decisions", i, len(decision_rows))

    bt = pd.DataFrame(rows)
    bt["forecast_correct"] = np.sign(bt["forecast_7d_pct"]) == np.sign(bt["realised_7d_pct"])
    # A directive's return: long on BUY, short on SELL, flat on HOLD, sized.
    side = bt["directive"].map({"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0})
    bt["strategy_return_pct"] = side * bt["realised_7d_pct"] * (bt["position_pct"] / 100)

    if save:
        bt.to_csv(BACKTEST_PATH, index=False)
    return bt


def summarise_backtest(bt: pd.DataFrame) -> dict:
    traded = bt[bt["directive"] != "HOLD"]
    summary = {
        "n_decisions": int(len(bt)),
        "directive_distribution": bt["directive"].value_counts().to_dict(),
        "agreement_distribution": bt["agreement"].value_counts().to_dict(),
        "traps_flagged": int(bt["trap_flagged"].sum()),
        "forecast_directional_accuracy": float(bt["forecast_correct"].mean()),
        "n_traded": int(len(traded)),
        "traded_hit_rate": float(
            (np.sign(traded["forecast_7d_pct"]) == np.sign(traded["realised_7d_pct"])).mean()
        ) if len(traded) else None,
        "mean_strategy_return_pct": float(bt["strategy_return_pct"].mean()),
        "buy_and_hold_mean_7d_pct": float(bt["realised_7d_pct"].mean()),
    }
    return summary


def run_all(save: bool = True) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ablation = forecaster_ablation(save=save)
    sentiment = sentiment_validation(save=save)
    thresholds = calibrate_decision_thresholds(save=save)
    bt = committee_backtest(save=save)
    summary = {
        "forecaster_ablation": ablation.to_dict("records"),
        "sentiment_validation": sentiment,
        "decision_thresholds": thresholds,
        "committee_backtest": summarise_backtest(bt),
    }
    if save:
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    out = run_all()
    print(json.dumps(out["committee_backtest"], indent=2, default=str))
