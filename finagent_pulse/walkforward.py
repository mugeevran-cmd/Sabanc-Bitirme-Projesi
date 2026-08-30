"""Walk-forward evaluation across market regimes.

The shipped split trains through 2022-06-30 and tests on 2023-04 → 2024-02, an
11-month bull run. Two limitations follow from that and neither can be answered
inside it: the committee issued **zero SELL directives**, so the short side is
untested, and every number in section 6 describes one regime.

This module refits the model on several chronological splits so each fold's test
window lands on a different market, and scores each with its own model. The
folds are chosen from the data rather than the calendar:

    2018  -3.8%   max drawdown -19.8%   (Q4 selloff)
    2019 +28.7%   max drawdown  -6.8%
    2020 +15.3%   max drawdown -33.9%   (COVID crash and recovery)
    2021 +28.8%   max drawdown  -5.2%
    2022 -20.0%   max drawdown -25.4%   (bear market)
    2023 +24.7%   max drawdown -10.3%

The 2022 fold is the point of the exercise: if the committee never issues a SELL
even there, that is a property of the decision rule, not of the test period.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from finagent_pulse import config
from finagent_pulse.data.preprocess import merge_features
from finagent_pulse.models import forecaster

log = logging.getLogger(__name__)

RESULTS_CSV = config.REPORTS / "walkforward.csv"
RESULTS_JSON = config.REPORTS / "walkforward.json"
DECISIONS_CSV = config.REPORTS / "walkforward_decisions.csv"

# (name, train_end, val_end, test_end, what the test window is)
FOLDS = [
    ("covid_2020", "2019-06-30", "2019-12-31", "2020-12-31",
     "COVID crash and recovery, -33.9% peak-to-trough"),
    ("bull_2021", "2020-06-30", "2020-12-31", "2021-12-31",
     "calm bull market, +28.8%"),
    ("bear_2022", "2021-06-30", "2021-12-31", "2022-12-31",
     "bear market, -20.0% with a -25.4% drawdown"),
    ("bull_2023", "2022-06-30", "2023-03-31", None,
     "the shipped split: 11-month bull run"),
]


def fold_config(train_end: str, val_end: str, test_end: str | None) -> config.LSTMConfig:
    return replace(config.LSTM, train_end=train_end, val_end=val_end, test_end=test_end)


def directional_edge(decisions: pd.DataFrame) -> dict:
    """Compare the model's directional calls against always predicting "up".

    Directional accuracy on its own flatters any model on an index that rises
    most weeks. The number that matters is the margin over a constant "up"
    call, because that is the effort-free baseline a 7-day equity forecast has
    to beat before it can be said to have direction in it at all.
    """
    predicted = np.sign(decisions["forecast_7d_pct"])
    realised = np.sign(decisions["realised_7d_pct"])
    model = float((predicted == realised).mean())
    always_up = float((realised > 0).mean())
    return {
        "share_forecast_negative": float((decisions["forecast_7d_pct"] < 0).mean()),
        "mean_forecast_pct": float(decisions["forecast_7d_pct"].mean()),
        "mean_realised_pct": float(decisions["realised_7d_pct"].mean()),
        "decision_day_accuracy": model,
        "always_up_accuracy": always_up,
        "edge_over_always_up": model - always_up,
    }


def run_fold(name: str, cfg: config.LSTMConfig, df: pd.DataFrame,
             description: str, backtest: bool = True) -> tuple[dict, pd.DataFrame]:
    """Fit and score one fold. Returns ``(summary, decisions)``."""
    from finagent_pulse.agents import committee

    ckpt = config.MODELS_OUT / f"walkforward_{name}.pt"
    log.info("[%s] training: train<=%s val<=%s test<=%s",
             name, cfg.train_end, cfg.val_end, cfg.test_end or "end")
    metrics = forecaster.train(df, cfg=cfg, tag=name, ckpt_path=ckpt)
    test = metrics["test"]

    summary = {
        "fold": name,
        "regime": description,
        "train_end": cfg.train_end,
        "val_end": cfg.val_end,
        "test_range": " → ".join(metrics["split_dates"]["test_range"]),
        "train_windows": int(metrics["validation"]["n_samples"] + test["n_samples"]),
        "test_windows": int(test["n_samples"]),
        "rmse_return": test["rmse_return_overall"],
        "r2_return": test["r2_return_overall"],
        "directional_acc_h7": test["directional_accuracy_h7"],
        "skill_vs_naive_pct": test["skill_vs_naive_pct"],
    }

    decisions = pd.DataFrame()
    if backtest:
        # Score this fold with the model that never saw it.
        committee.set_forecast_service(forecaster.ForecastService(ckpt_path=ckpt))
        try:
            from finagent_pulse.evaluation import committee_backtest, summarise_backtest
            decisions = committee_backtest(save=False, start=cfg.val_end,
                                           end=cfg.test_end)
            decisions.insert(0, "fold", name)
            stats = summarise_backtest(decisions)
            counts = stats["directive_distribution"]
            summary.update({
                "n_decisions": stats["n_decisions"],
                "n_buy": counts.get("BUY", 0),
                "n_sell": counts.get("SELL", 0),
                "n_hold": counts.get("HOLD", 0),
                "traded_hit_rate": stats["traded_hit_rate"],
                "mean_strategy_return_pct": stats["mean_strategy_return_pct"],
                "buy_and_hold_mean_7d_pct": stats["buy_and_hold_mean_7d_pct"],
                **directional_edge(decisions),
            })
        finally:
            committee.set_forecast_service(None)

    log.info("[%s] DA=%.3f skill=%+.2f%% directives: %s",
             name, summary["directional_acc_h7"], summary["skill_vs_naive_pct"],
             {k: summary.get(k) for k in ("n_buy", "n_sell", "n_hold")})
    return summary, decisions


def run_all(backtest: bool = True, save: bool = True) -> pd.DataFrame:
    df = merge_features()
    rows, frames = [], []
    for name, train_end, val_end, test_end, description in FOLDS:
        cfg = fold_config(train_end, val_end, test_end)
        summary, decisions = run_fold(name, cfg, df, description, backtest=backtest)
        rows.append(summary)
        if len(decisions):
            frames.append(decisions)

    table = pd.DataFrame(rows)
    if save:
        table.to_csv(RESULTS_CSV, index=False)
        RESULTS_JSON.write_text(json.dumps(
            {"folds": table.to_dict("records"),
             "note": ("Each fold is fitted and scored on its own chronological "
                      "split; no fold's model has seen its own test window.")},
            indent=2, default=str))
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(DECISIONS_CSV, index=False)
    return table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    out = run_all()
    cols = ["fold", "test_range", "directional_acc_h7", "skill_vs_naive_pct",
            "n_buy", "n_sell", "n_hold", "decision_day_accuracy",
            "always_up_accuracy", "edge_over_always_up"]
    print("\n" + out[[c for c in cols if c in out]].round(4).to_string(index=False))
