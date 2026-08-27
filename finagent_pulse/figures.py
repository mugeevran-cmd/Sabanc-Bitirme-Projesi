"""Generate the static figures used in the technical report."""
from __future__ import annotations

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finagent_pulse import config
from finagent_pulse.data.preprocess import merge_features

log = logging.getLogger(__name__)

FIG_DIR = config.REPORTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})

BLUE, GREEN, RED, GREY = "#2E86DE", "#26A69A", "#EF5350", "#8892A0"


def fig_overview() -> None:
    """Price, sentiment and news coverage across the whole study period."""
    df = merge_features()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2, 1]})

    axes[0].plot(df["date"], df["close"], color=BLUE, lw=1.2)
    axes[0].set_ylabel("Index level")
    axes[0].set_title("S&P 500 with FinBERT news sentiment, 2018-2024")

    for boundary, label in ((config.LSTM.train_end, "train│val"),
                            (config.LSTM.val_end, "val│test")):
        for ax in axes:
            ax.axvline(pd.Timestamp(boundary), color=GREY, ls="--", lw=0.9)
        axes[0].annotate(label, (pd.Timestamp(boundary), axes[0].get_ylim()[1]),
                         fontsize=7, color=GREY, ha="center", va="bottom")

    s = df["sent_mean"].rolling(10, min_periods=1).mean()
    axes[1].fill_between(df["date"], 0, s, where=s >= 0, color=GREEN, alpha=0.75, lw=0)
    axes[1].fill_between(df["date"], 0, s, where=s < 0, color=RED, alpha=0.75, lw=0)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel("Sentiment\n(10d mean)")

    axes[2].bar(df["date"], df["headline_count"], color=GREY, width=2.0)
    axes[2].set_ylabel("Headlines\nper session")
    axes[2].set_xlabel("Date")

    fig.savefig(FIG_DIR / "01_overview.png")
    plt.close(fig)


def fig_forecast_quality() -> None:
    """Per-horizon error and directional accuracy, plus predicted vs realised."""
    metrics = json.loads((config.MODELS_OUT / "forecast_metrics.json").read_text())
    per_h = pd.DataFrame(metrics["test"]["per_horizon"])
    preds = pd.read_parquet(config.MODELS_OUT / "test_predictions.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    ax.bar(per_h["horizon"], per_h["rmse_return"], color=BLUE, alpha=0.85)
    ax.set_xlabel("Horizon (sessions)")
    ax.set_ylabel("RMSE (log-return)")
    ax.set_title("Error grows with horizon")

    ax = axes[1]
    ax.plot(per_h["horizon"], per_h["directional_accuracy"], "o-", color=GREEN, lw=2)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.text(1.1, 0.505, "coin flip", fontsize=7, color=GREY)
    ax.set_ylim(0.35, 0.85)
    ax.set_xlabel("Horizon (sessions)")
    ax.set_ylabel("Directional accuracy")
    ax.set_title("Direction improves with horizon")

    ax = axes[2]
    h = config.LSTM.horizon
    ax.scatter(preds[f"true_h{h}"] * 100, preds[f"pred_h{h}"] * 100,
               s=12, alpha=0.55, color=BLUE, edgecolor="none")
    lim = float(np.abs(preds[f"true_h{h}"] * 100).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=GREY, ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Realised 7-day return (%)")
    ax.set_ylabel("Predicted (%)")
    ax.set_title("Predictions are correctly signed\nbut heavily shrunk")

    fig.savefig(FIG_DIR / "02_forecast_quality.png")
    plt.close(fig)


def fig_rag_ablation() -> None:
    rag = pd.read_csv(config.REPORTS / "rag_ablation.csv")
    styles = [s for s in rag["query_style"].unique() if s != "macro_average"]
    modes = ["bm25", "vector", "hybrid", "hybrid_kg"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, title in zip(axes, ["ndcg@k", "recall@k"],
                                 ["nDCG@10", "Recall@10"]):
        width = 0.35
        x = np.arange(len(modes))
        for i, style in enumerate(styles):
            sub = rag[rag["query_style"] == style].set_index("mode").loc[modes]
            ax.bar(x + (i - 0.5) * width, sub[metric], width,
                   label=f"{style} queries",
                   color=[BLUE, GREEN][i], alpha=0.9)
        macro = rag[rag["query_style"] == "macro_average"].set_index("mode").loc[modes]
        ax.plot(x, macro[metric], "k_", markersize=22, mew=2, label="macro average")
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=12)
        ax.set_ylabel(title)
        ax.set_title(f"{title} by retrieval mode")
    axes[0].legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Hybrid retrieval buys robustness, not a higher peak",
                 fontsize=10.5, fontweight="bold", y=1.03)
    fig.savefig(FIG_DIR / "03_rag_ablation.png")
    plt.close(fig)


def fig_sentiment_validation() -> None:
    df = merge_features()
    test = df[df["date"] > pd.Timestamp(config.LSTM.val_end)].copy()
    test["next_return"] = np.log(test["close"]).diff().shift(-1)
    v = test.dropna(subset=["next_return", "sent_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in (
        (axes[0], "log_return", "Same day: strong relationship"),
        (axes[1], "next_return", "Next day: none"),
    ):
        ax.scatter(v["sent_mean"], v[col] * 100, s=14, alpha=0.5,
                   color=BLUE, edgecolor="none")
        m, b = np.polyfit(v["sent_mean"], v[col] * 100, 1)
        xs = np.linspace(v["sent_mean"].min(), v["sent_mean"].max(), 50)
        ax.plot(xs, m * xs + b, color=RED, lw=2)
        r = float(np.corrcoef(v["sent_mean"], v[col])[0, 1])
        ax.set_title(f"{title}  (r = {r:+.3f})")
        ax.set_xlabel("Daily FinBERT sentiment")
        ax.set_ylabel("Return (%)")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)

    fig.suptitle("Headline sentiment describes the session it belongs to, "
                 "and does not forecast the next one",
                 fontsize=10.5, fontweight="bold", y=1.04)
    fig.savefig(FIG_DIR / "04_sentiment_validation.png")
    plt.close(fig)


def fig_backtest() -> None:
    bt = pd.read_csv(config.REPORTS / "committee_backtest.csv", parse_dates=["date"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8),
                             gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(bt["date"], bt["close"], color=GREY, lw=1.2, label="S&P 500")
    traded = bt[bt["directive"] != "HOLD"]
    for directive, color, marker in (("BUY", GREEN, "^"), ("SELL", RED, "v")):
        sel = traded[traded["directive"] == directive]
        if len(sel):
            ax.scatter(sel["date"], sel["close"], color=color, marker=marker,
                       s=70, zorder=5, label=f"{directive} ({len(sel)})",
                       edgecolor="white", linewidth=0.8)
    traps = bt[bt["trap_flagged"]]
    if len(traps):
        ax.scatter(traps["date"], traps["close"], facecolors="none",
                   edgecolors="#F9A825", s=110, lw=1.6, zorder=4,
                   label=f"trap flagged ({len(traps)})")
    ax.set_title("Committee decisions over the held-out test period")
    ax.set_ylabel("Index level")
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[1]
    counts = bt["directive"].value_counts()
    ax.bar(counts.index, counts.values,
           color=[{"HOLD": GREY, "BUY": GREEN, "SELL": RED}[i] for i in counts.index])
    for i, (label, value) in enumerate(counts.items()):
        ax.text(i, value + 0.7, f"{value}\n({value / len(bt):.0%})",
                ha="center", fontsize=8)
    ax.set_ylim(0, counts.max() * 1.28)
    ax.set_ylabel("Decisions")
    ax.set_title("The committee mostly abstains")

    fig.savefig(FIG_DIR / "05_backtest.png")
    plt.close(fig)


def fig_knowledge_graph(top_n: int = 30) -> None:
    import pickle

    import networkx as nx

    with open(config.KG_PATH, "rb") as fh:
        g = pickle.load(fh)

    keep = [n for n, _ in sorted(g.degree(weight="weight"),
                                 key=lambda kv: -kv[1])[:top_n]]
    sub = g.subgraph(keep)
    # Show only the strongest edges, otherwise the layout is an unreadable hairball.
    weights = sorted((d["weight"] for _a, _b, d in sub.edges(data=True)), reverse=True)
    cutoff = weights[min(len(weights) - 1, 70)]
    edges = [(a, b, d) for a, b, d in sub.edges(data=True) if d["weight"] >= cutoff]

    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(sub, seed=7, k=1.6, iterations=300)

    palette = {"company": BLUE, "index": "#8E44AD", "macro": GREEN,
               "institution": "#E67E22", "regime": RED, "asset": "#C2A000"}
    sizes = [80 + 9 * np.sqrt(sub.nodes[n].get("mentions", 1)) for n in sub.nodes]
    colors = [palette.get(sub.nodes[n].get("entity_type"), GREY) for n in sub.nodes]

    nx.draw_networkx_edges(sub, pos, edgelist=[(a, b) for a, b, _ in edges],
                           width=[0.4 + 2.2 * d["weight"] / weights[0] for _a, _b, d in edges],
                           alpha=0.28, edge_color=GREY, ax=ax)
    nx.draw_networkx_nodes(sub, pos, node_size=sizes, node_color=colors,
                           alpha=0.9, linewidths=0.5, edgecolors="white", ax=ax)
    nx.draw_networkx_labels(sub, pos, font_size=7.5, ax=ax,
                            bbox=dict(facecolor="white", alpha=0.65,
                                      edgecolor="none", pad=0.6))

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t, markersize=7)
               for t, c in palette.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper left", ncol=2)
    ax.set_title(f"Entity co-occurrence knowledge graph "
                 f"(top {top_n} entities, strongest edges)", fontweight="bold")
    ax.axis("off")

    fig.savefig(FIG_DIR / "06_knowledge_graph.png")
    plt.close(fig)


def build_all() -> None:
    for fn in (fig_overview, fig_forecast_quality, fig_rag_ablation,
               fig_sentiment_validation, fig_backtest, fig_knowledge_graph):
        fn()
        log.info("wrote %s", fn.__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_all()
    print("\n".join(sorted(p.name for p in FIG_DIR.glob("*.png"))))
