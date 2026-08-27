"""FinAgent-Pulse — interactive Streamlit dashboard.

Run with:  streamlit run finagent_pulse/app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finagent_pulse import config                                    # noqa: E402
from finagent_pulse.data.preprocess import merge_features            # noqa: E402
from finagent_pulse.models.sentiment import fear_greed_index         # noqa: E402

st.set_page_config(page_title="FinAgent-Pulse", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

ACCENT = "#2E86DE"
UP, DOWN, MUTED = "#26A69A", "#EF5350", "#8892A0"


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    df = merge_features()
    df["fear_greed"] = fear_greed_index(df)
    return df


@st.cache_data(show_spinner=False)
def load_headlines() -> pd.DataFrame:
    return pd.read_parquet(config.HEADLINES_SCORED)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource(show_spinner="Loading retrieval indexes…")
def get_retriever_cached():
    from finagent_pulse.rag.hybrid import get_retriever
    return get_retriever()


@st.cache_resource(show_spinner="Loading forecaster…")
def get_forecaster_cached():
    from finagent_pulse.models.forecaster import ForecastService
    return ForecastService()


@st.cache_data(show_spinner="Convening the investment committee…")
def run_committee_cached(as_of: str) -> dict:
    from finagent_pulse.agents.committee import executive_report, run_committee
    features = load_features()
    state = run_committee(features, as_of)
    return {
        "quant": state["quant"],
        "sentiment": state["sentiment"],
        "risk": state["risk"],
        "report": executive_report(state),
    }


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def price_and_forecast_chart(df: pd.DataFrame, as_of: pd.Timestamp,
                             forecast: dict | None, lookback: int = 180) -> go.Figure:
    hist = df[df["date"] <= as_of].tail(lookback)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.06,
                        subplot_titles=(f"{config.TICKER_LABEL} — price & 7-day forecast",
                                        "Daily FinBERT sentiment"))

    fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"], name="Close",
                             line=dict(color=ACCENT, width=2)), row=1, col=1)

    if forecast:
        # Project onto the next business days after the decision date.
        future = pd.bdate_range(as_of + pd.Timedelta(days=1),
                                periods=config.LSTM.horizon)
        path = [forecast["origin_close"]] + list(forecast["prices"])
        xs = [as_of] + list(future)
        up = forecast["prices"][-1] >= forecast["origin_close"]
        fig.add_trace(go.Scatter(
            x=xs, y=path, name="Bi-LSTM forecast",
            line=dict(color=UP if up else DOWN, width=2.5, dash="dot"),
            mode="lines+markers", marker=dict(size=5)), row=1, col=1)

        # Uncertainty band: +/-1 horizon volatility around the projected path.
        vol = float(df.loc[df["date"] == as_of, "volatility_20d"].iloc[0])
        steps = np.sqrt(np.arange(1, config.LSTM.horizon + 1))
        band = np.array(forecast["prices"]) * vol * steps
        fig.add_trace(go.Scatter(
            x=list(future) + list(future)[::-1],
            y=list(np.array(forecast["prices"]) + band) +
              list((np.array(forecast["prices"]) - band)[::-1]),
            fill="toself", fillcolor="rgba(46,134,222,0.13)",
            line=dict(width=0), name="±1σ band", hoverinfo="skip"), row=1, col=1)

    colors = [UP if s > 0.05 else DOWN if s < -0.05 else MUTED
              for s in hist["sent_mean"]]
    fig.add_trace(go.Bar(x=hist["date"], y=hist["sent_mean"], name="Sentiment",
                         marker_color=colors, showlegend=False), row=2, col=1)

    fig.update_layout(height=560, hovermode="x unified", template="plotly_white",
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_yaxes(title_text="Index level", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", range=[-1, 1], row=2, col=1)
    return fig


def fear_greed_gauge(value: float) -> go.Figure:
    if value < 25:
        label, color = "Extreme Fear", "#C62828"
    elif value < 45:
        label, color = "Fear", "#EF6C00"
    elif value < 55:
        label, color = "Neutral", "#9E9E9E"
    elif value < 75:
        label, color = "Greed", "#66BB6A"
    else:
        label, color = "Extreme Greed", "#2E7D32"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": f"Market Fear & Greed<br><span style='font-size:0.8em'>{label}</span>"},
        number={"suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.75},
            "steps": [
                {"range": [0, 25], "color": "#FFEBEE"},
                {"range": [25, 45], "color": "#FFF3E0"},
                {"range": [45, 55], "color": "#F5F5F5"},
                {"range": [55, 75], "color": "#E8F5E9"},
                {"range": [75, 100], "color": "#C8E6C9"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
def tab_dashboard(df: pd.DataFrame, as_of: pd.Timestamp) -> None:
    row = df[df["date"] == as_of].iloc[0]

    try:
        forecast = get_forecaster_cached().forecast(df, as_of=as_of)
    except Exception as exc:
        forecast = None
        st.warning(f"Forecast unavailable: {exc}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Close", f"{row['close']:,.2f}", f"{(np.exp(row['log_return']) - 1) * 100:+.2f}%")
    c2.metric("7-day forecast",
              f"{forecast['total_return_pct']:+.2f}%" if forecast else "n/a")
    c3.metric("Sentiment", f"{row['sent_mean']:+.3f}",
              f"{row['sent_mean'] - row['sent_mean_5d']:+.3f} vs 5d")
    c4.metric("Headlines", f"{int(row['headline_count'])}",
              f"{row['headline_count_z']:+.1f}σ")
    c5.metric("20d volatility", f"{row['volatility_20d'] * 100:.2f}%")

    left, right = st.columns([3, 1])
    left.plotly_chart(price_and_forecast_chart(df, as_of, forecast),
                      width="stretch")
    with right:
        st.plotly_chart(fear_greed_gauge(float(row["fear_greed"])),
                        width="stretch")
        st.caption("Sentiment percentile-ranked across the full 2018-2024 study "
                   "period, so 50 means a typical day rather than an absolute "
                   "neutral level.")

    if forecast:
        st.subheader("Projected trajectory")
        traj = pd.DataFrame({
            "Session": [f"t+{i}" for i in range(1, config.LSTM.horizon + 1)],
            "Projected level": [f"{p:,.2f}" for p in forecast["prices"]],
            "Cumulative return": [f"{r:+.2f}%" for r in forecast["path_return_pct"]],
        })
        st.dataframe(traj, width="stretch", hide_index=True)


def tab_committee(as_of: pd.Timestamp) -> None:
    st.caption("Three agents run in sequence via LangGraph. Findings and the "
               "final directive are computed deterministically; the language "
               "model only writes the explanation.")
    if not st.button("▶︎ Convene the investment committee", type="primary"):
        st.info("Press the button to run the committee for the selected date.")
        return

    result = run_committee_cached(str(as_of.date()))
    risk = result["risk"]

    color = {"BUY": UP, "SELL": DOWN, "HOLD": MUTED}[risk["directive"]]
    st.markdown(
        f"<div style='padding:1rem 1.25rem;border-left:6px solid {color};"
        f"background:rgba(0,0,0,0.03);border-radius:6px'>"
        f"<h2 style='margin:0;color:{color}'>{risk['directive']}</h2>"
        f"<p style='margin:0.35rem 0 0'>Position size <b>{risk['position_pct']:.1f}%</b> "
        f"of standard · conviction <b>{risk['conviction']:.0%}</b> · signals "
        f"<b>{risk['agreement'].replace('_', ' ')}</b></p></div>",
        unsafe_allow_html=True)

    if risk["trap_warning"]:
        st.warning(risk["trap_warning"].replace("**", ""))

    st.markdown("---")
    st.markdown(result["report"])
    st.download_button("Download report (markdown)", result["report"],
                       file_name=f"finagent_report_{as_of.date()}.md")


def tab_rag() -> None:
    st.caption("Compare the four retrieval modes on the same query. This is the "
               "ablation study made interactive.")
    retriever = get_retriever_cached()

    query = st.text_input("Query",
                          "What risks are investors reacting to right now?")
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox("Retrieval mode",
                        ["hybrid_kg", "hybrid", "vector", "bm25"])
    start = c2.date_input("From", pd.Timestamp("2023-01-01"))
    end = c3.date_input("To", pd.Timestamp("2023-12-31"))

    if st.button("Search", type="primary"):
        docs, diag = retriever.retrieve(query, mode=mode, top_k=10,
                                        start=str(start), end=str(end))
        if diag["kg_neighbours"]:
            st.info("Knowledge-graph expansion added: "
                    + ", ".join(f"`{n}`" for n in diag["kg_neighbours"]))
        if diag["arm_weights"]:
            st.caption(f"Fusion arm weights: {diag['arm_weights']}")

        if not docs:
            st.warning("No documents matched.")
            return
        st.dataframe(pd.DataFrame([{
            "Date": d.date, "Headline": d.headline, "Sentiment": round(d.sentiment, 3),
            "Label": d.label, "Entities": ", ".join(d.entities),
            "Retrieved by": d.provenance,
        } for d in docs]), width="stretch", hide_index=True)


def tab_evaluation() -> None:
    fm = load_json(config.MODELS_OUT / "forecast_metrics.json")
    if fm:
        st.subheader("Bi-LSTM forecaster — held-out test period")
        t = fm["test"]
        st.caption(f"Test window {fm['split_dates']['test_range'][0]} → "
                   f"{fm['split_dates']['test_range'][1]} · {t['n_samples']} windows, "
                   "never seen during training or model selection.")
        c = st.columns(5)
        c[0].metric("RMSE (return)", f"{t['rmse_return_overall']:.5f}")
        c[1].metric("R² (return)", f"{t['r2_return_overall']:+.4f}")
        c[2].metric("R² (price)", f"{t['r2_price_overall']:.4f}")
        c[3].metric("MAPE (price)", f"{t['mape_price_pct']:.2f}%")
        c[4].metric("Direction @ h=7", f"{t['directional_accuracy_h7']:.1%}")

        st.info(
            "**Read R² in return space, not price space.** The 0.93 price-level "
            "R² mostly reflects that next week's index is close to this week's; "
            "any naive model scores similarly. The return-space R² near zero is "
            "the honest measure, and is what short-horizon index forecasting "
            "actually looks like. The meaningful result is directional accuracy.")

        per_h = pd.DataFrame(t["per_horizon"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=per_h["horizon"], y=per_h["rmse_return"],
                             name="RMSE", marker_color=ACCENT), secondary_y=False)
        fig.add_trace(go.Scatter(x=per_h["horizon"], y=per_h["directional_accuracy"],
                                 name="Directional accuracy", mode="lines+markers",
                                 line=dict(color=UP, width=3)), secondary_y=True)
        fig.add_hline(y=0.5, line_dash="dash", line_color=MUTED, secondary_y=True)
        fig.update_layout(height=340, template="plotly_white",
                          title="Error and directional accuracy by horizon",
                          margin=dict(l=10, r=10, t=50, b=10))
        fig.update_xaxes(title_text="Forecast horizon (sessions)")
        fig.update_yaxes(title_text="RMSE (log-return)", secondary_y=False)
        fig.update_yaxes(title_text="Directional accuracy", range=[0.3, 0.85],
                         secondary_y=True)
        st.plotly_chart(fig, width="stretch")

    ab = load_csv(config.REPORTS / "forecaster_ablation.csv")
    if ab is not None:
        st.subheader("Does the sentiment channel help?")
        st.dataframe(ab.round(5), width="stretch", hide_index=True)

    rag = load_csv(config.REPORTS / "rag_ablation.csv")
    if rag is not None:
        st.subheader("Standard RAG vs Hybrid RAG")
        st.dataframe(rag.round(4), width="stretch", hide_index=True)
        macro = rag[rag["query_style"] != "macro_average"]
        fig = go.Figure()
        for style in macro["query_style"].unique():
            sub = macro[macro["query_style"] == style]
            fig.add_trace(go.Bar(x=sub["mode"], y=sub["ndcg@k"], name=f"{style} queries"))
        fig.update_layout(height=340, template="plotly_white", barmode="group",
                          title="nDCG@10 by retrieval mode and query style",
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("BM25 wins on keyword queries and collapses on natural-language "
                   "ones; the dense index does the reverse. The fused modes are the "
                   "only ones that stay strong on both — robustness, not a bigger "
                   "peak, is what hybrid retrieval buys.")

    sv = load_json(config.REPORTS / "sentiment_validation.json")
    if sv:
        st.subheader("Sentiment engine validation")
        c = st.columns(4)
        c[0].metric("Same-day correlation", f"{sv['corr_same_day']['r']:+.3f}",
                    f"p={sv['corr_same_day']['p_value']:.1e}")
        c[1].metric("Next-day correlation", f"{sv['corr_next_day']['r']:+.3f}",
                    f"p={sv['corr_next_day']['p_value']:.2f}")
        c[2].metric("Next-day hit rate", f"{sv['next_day_hit_rate']:.1%}")
        c[3].metric("Sessions", sv["n_sessions"])
        st.caption("FinBERT tracks *what already happened* very strongly and "
                   "predicts the next session essentially not at all. That is the "
                   "expected result for published headlines, and it is why the "
                   "committee treats sentiment as confirmation rather than as a "
                   "forecast in its own right.")

    bt = load_csv(config.REPORTS / "committee_backtest.csv")
    if bt is not None:
        st.subheader("Committee backtest")
        traded = bt[bt["directive"] != "HOLD"]
        c = st.columns(4)
        c[0].metric("Decisions", len(bt))
        c[1].metric("Traded", f"{len(traded)} ({len(traded) / len(bt):.0%})")
        if len(traded):
            hit = (np.sign(traded["forecast_7d_pct"]) ==
                   np.sign(traded["realised_7d_pct"])).mean()
            c[2].metric("Hit rate when traded", f"{hit:.1%}")
        c[3].metric("Traps flagged", int(bt["trap_flagged"].sum()))
        st.dataframe(bt, width="stretch", hide_index=True)


def tab_corpus(df: pd.DataFrame) -> None:
    hl = load_headlines()
    c = st.columns(4)
    c[0].metric("Headlines", f"{len(hl):,}")
    c[1].metric("Trading sessions", f"{len(df):,}")
    c[2].metric("Period", f"{hl['date'].min():%Y-%m} → {hl['date'].max():%Y-%m}")
    c[3].metric("Mean sentiment", f"{hl['sentiment'].mean():+.3f}")

    yearly = hl.groupby(hl["date"].dt.year).agg(
        headlines=("headline", "size"), mean_sentiment=("sentiment", "mean")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=yearly["date"], y=yearly["headlines"], name="Headlines",
                         marker_color=ACCENT), secondary_y=False)
    fig.add_trace(go.Scatter(x=yearly["date"], y=yearly["mean_sentiment"],
                             name="Mean sentiment", mode="lines+markers",
                             line=dict(color=DOWN, width=3)), secondary_y=True)
    fig.update_layout(height=340, template="plotly_white",
                      title="Corpus coverage and sentiment by year",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("2024 covers January–March only: the source corpus ends 2024-03-04.")

    st.subheader("Browse headlines")
    day = st.date_input("Date", hl["date"].max(),
                        min_value=hl["date"].min(), max_value=hl["date"].max())
    sel = hl[hl["date"] == pd.Timestamp(day)]
    if sel.empty:
        st.info("No headlines on that date (weekend, holiday, or a gap in coverage).")
    else:
        st.dataframe(
            sel[["headline", "sentiment", "label", "confidence"]]
            .sort_values("sentiment", ascending=False).round(3),
            width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    st.title("📈 FinAgent-Pulse")
    st.caption("Multi-agent quantitative trading & sentiment analysis using "
               "Hybrid RAG and time-series forecasting — S&P 500, 2018–2024")

    try:
        df = load_features()
    except FileNotFoundError:
        st.error("Processed data not found. Run `python -m finagent_pulse.pipeline` first.")
        return

    with st.sidebar:
        st.header("Decision date")
        as_of = st.date_input(
            "As of", df["date"].max().date(),
            min_value=df["date"].iloc[config.LSTM.lookback].date(),
            max_value=df["date"].max().date())
        as_of = pd.Timestamp(as_of)
        if as_of not in set(df["date"]):
            prior = df.loc[df["date"] <= as_of, "date"]
            if prior.empty:
                st.error("No trading session on or before that date.")
                return
            as_of = prior.iloc[-1]
            st.info(f"Snapped to the previous session: {as_of.date()}")

        st.markdown("---")
        st.markdown(
            f"**Corpus** {len(load_headlines()):,} headlines  \n"
            f"**Sessions** {len(df):,}  \n"
            f"**Ticker** {config.TICKER_LABEL}  \n"
            f"**Horizon** {config.LSTM.horizon} sessions")
        st.markdown("---")
        st.caption("Academic prototype. Not investment advice.")

    t1, t2, t3, t4, t5 = st.tabs(
        ["Dashboard", "Investment Committee", "Hybrid RAG", "Evaluation", "Corpus"])
    with t1:
        tab_dashboard(df, as_of)
    with t2:
        tab_committee(as_of)
    with t3:
        tab_rag()
    with t4:
        tab_evaluation()
    with t5:
        tab_corpus(df)


if __name__ == "__main__":
    main()
