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
# Chart presentation
#
# One place for the settings that decide whether a chart can be read at a
# glance, so the five figures in this app agree with each other instead of each
# carrying its own font sizes. Nothing here changes what is plotted.
# --------------------------------------------------------------------------
GRID = "rgba(136,146,160,0.20)"

# Legends sat at y=1.08 -- directly on top of the subplot titles, which made
# both unreadable. They now sit below the plot, centred, in a size that can
# actually be read, and never compete with a title for the same pixels.
LEGEND_BELOW = dict(
    orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5,
    font=dict(size=13), bgcolor="rgba(0,0,0,0)", itemsizing="constant",
    itemwidth=40,
)

# Same idea, dropped far enough to clear an x-axis title. A chart that labels
# its x-axis needs the extra room; one that does not would just get a gap.
LEGEND_BELOW_AXIS = {**LEGEND_BELOW, "y": -0.30}

def style_fig(fig: go.Figure, height: int, legend: dict | None = None,
              title: str | None = None) -> go.Figure:
    """Apply the shared look: readable type, quiet gridlines, room to breathe."""
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15), x=0,
                                     xanchor="left", y=0.97, yanchor="top"))
    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(size=13),
        hoverlabel=dict(font_size=13, namelength=-1),
        legend=legend or LEGEND_BELOW,
        margin=dict(l=10, r=10, t=60 if title else 40,
                    b=90 if legend is LEGEND_BELOW_AXIS else 60),
        bargap=0.15,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=12),
                     title_font=dict(size=13))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=12),
                     title_font=dict(size=13))
    # Subplot titles are annotations, not layout.title, so they need sizing of
    # their own -- otherwise they render at the Plotly default and fight the
    # axis labels for attention.
    for note in fig.layout.annotations or ():
        note.font.size = 14
    return fig


CSS = """
<style>
  /* Bigger than the default, but not so big that a long value -- the Corpus
     tab's "2018-01 -> 2024-03" is the widest one -- gets ellipsised. */
  [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; }
  [data-testid="stMetricLabel"] p { font-size: 0.82rem; letter-spacing: .04em;
                                    text-transform: uppercase; opacity: .72; }
  [data-testid="stMetricDelta"] { font-size: 0.85rem; }
  /* Tabs as a real navigation bar rather than four words in a row. */
  .stTabs [data-baseweb="tab-list"] { gap: .35rem; }
  .stTabs [data-baseweb="tab"] { padding: .55rem 1.05rem; font-size: .95rem;
                                 font-weight: 500; }
  /* Dataframes sit flush against their captions otherwise. */
  [data-testid="stDataFrame"] { margin-top: .35rem; }
</style>
"""


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


@st.cache_resource(show_spinner="Loading forecaster…")
def get_forecaster_cached():
    from finagent_pulse.models.forecaster import ForecastService
    return ForecastService()


@st.cache_data(show_spinner="Convening the investment committee…")
def run_committee_cached(as_of: str) -> dict:
    from finagent_pulse.agents.committee import executive_report, run_committee
    from finagent_pulse.agents.llm import get_writer
    features = load_features()
    state = run_committee(features, as_of)
    writer = get_writer()
    return {
        "quant": state["quant"],
        "sentiment": state["sentiment"],
        "risk": state["risk"],
        "report": executive_report(state),
        # Captured here, not read later: this result is cached, so a question
        # asked after the fact would answer about the wrong run.
        "narrative_mode": writer.mode,
        "narrative_error": writer.last_error,
    }


# --------------------------------------------------------------------------
# Preflight
#
# Every tab depends on a different generated artefact. Checking them up front
# turns "ChromaDB collection not found" three frames deep into one sentence
# naming the missing folder and the command that builds it.
# --------------------------------------------------------------------------
ARTIFACTS: dict[str, tuple[Path, str]] = {
    "Processed features": (config.FEATURES, "every tab"),
    "Scored headlines": (config.HEADLINES_SCORED, "Corpus"),
    "Forecaster checkpoint": (config.MODELS_OUT / "bilstm.pt",
                              "forecast trajectory, Investment Committee"),
    "Search indexes": (config.CHROMA_DIR, "Investment Committee"),
}

BUILD_HINT = "Run `python -m finagent_pulse.pipeline`, or `./setup.sh` for a first-time setup."


def missing_artifacts() -> list[tuple[str, Path, str]]:
    return [(label, path, needed_by)
            for label, (path, needed_by) in ARTIFACTS.items() if not path.exists()]


def require(path: Path, what: str) -> bool:
    """Render a clear notice and return False when ``path`` is not built yet."""
    if path.exists():
        return True
    st.info(f"**{what}** is not available yet — `{path.name}` has not been built.\n\n"
            + BUILD_HINT)
    return False


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def price_and_forecast_chart(df: pd.DataFrame, as_of: pd.Timestamp,
                             forecast: dict | None, lookback: int = 180) -> go.Figure:
    hist = df[df["date"] <= as_of].tail(lookback)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.12,
                        subplot_titles=(f"{config.TICKER_LABEL} — price & 7-day forecast",
                                        "Daily FinBERT sentiment"))

    fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"], name="Close",
                             line=dict(color=ACCENT, width=2.2),
                             hovertemplate="%{y:,.2f}<extra>Close</extra>"),
                  row=1, col=1)

    if forecast:
        # Project onto the next business days after the decision date.
        future = pd.bdate_range(as_of + pd.Timedelta(days=1),
                                periods=config.LSTM.horizon)
        path = [forecast["origin_close"]] + list(forecast["prices"])
        xs = [as_of] + list(future)
        up = forecast["prices"][-1] >= forecast["origin_close"]
        fig.add_trace(go.Scatter(
            x=xs, y=path, name="Bi-LSTM forecast",
            line=dict(color=UP if up else DOWN, width=2.6, dash="dot"),
            mode="lines+markers", marker=dict(size=7, symbol="circle",
                                              line=dict(width=0)),
            hovertemplate="%{y:,.2f}<extra>Forecast</extra>"), row=1, col=1)

        # Uncertainty band: +/-1 horizon volatility around the projected path.
        vol = float(df.loc[df["date"] == as_of, "volatility_20d"].iloc[0])
        steps = np.sqrt(np.arange(1, config.LSTM.horizon + 1))
        band = np.array(forecast["prices"]) * vol * steps
        fig.add_trace(go.Scatter(
            x=list(future) + list(future)[::-1],
            y=list(np.array(forecast["prices"]) + band) +
              list((np.array(forecast["prices"]) - band)[::-1]),
            fill="toself", fillcolor="rgba(46,134,222,0.16)",
            line=dict(width=0), name="±1σ band", hoverinfo="skip"), row=1, col=1)

        # A marker on the decision date, so the eye finds where history stops
        # and the projection starts without hunting for the change of dash.
        fig.add_vline(x=as_of, line=dict(color=MUTED, width=1, dash="dash"),
                      row=1, col=1)

    colors = [UP if s > 0.05 else DOWN if s < -0.05 else MUTED
              for s in hist["sent_mean"]]
    fig.add_trace(go.Bar(x=hist["date"], y=hist["sent_mean"], name="Sentiment",
                         marker_color=colors, showlegend=False,
                         hovertemplate="%{y:+.3f}<extra>Sentiment</extra>"),
                  row=2, col=1)

    style_fig(fig, height=620)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="Index level", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", range=[-1, 1], dtick=0.5,
                     row=2, col=1)
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
        title={"text": f"Market Fear & Greed<br>"
                       f"<span style='font-size:0.75em;color:{color}'>{label}</span>",
               "font": {"size": 16}},
        number={"suffix": "/100", "font": {"size": 26, "color": color}},
        gauge={
            # Ticks were unlabelled apart from the two ends, and the right-hand
            # "100" was clipped. Naming the quartiles turns the dial into
            # something you can actually read a position off.
            "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100],
                     "tickfont": {"size": 11}, "tickwidth": 1,
                     "tickcolor": MUTED},
            "bar": {"color": color, "thickness": 0.55},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            # 50 is the definition's own reference point -- "a typical day so
            # far" -- so it gets a line rather than being left to the eye.
            "threshold": {"line": {"color": MUTED, "width": 2}, "value": 50,
                          "thickness": 0.85},
            "steps": [
                {"range": [0, 25], "color": "rgba(198,40,40,0.22)"},
                {"range": [25, 45], "color": "rgba(239,108,0,0.18)"},
                {"range": [45, 55], "color": "rgba(158,158,158,0.18)"},
                {"range": [55, 75], "color": "rgba(102,187,106,0.18)"},
                {"range": [75, 100], "color": "rgba(46,125,50,0.22)"},
            ],
        },
    ))
    # The step colours are translucent so the dial reads the same on a light
    # and a dark theme; the opaque pastels only worked on white.
    fig.update_layout(height=380, margin=dict(l=45, r=45, t=85, b=45),
                      font=dict(size=13), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _split_of(as_of: pd.Timestamp) -> str:
    """Which split a decision date falls in, using the forecaster's boundaries."""
    if as_of <= pd.Timestamp(config.LSTM.train_end):
        return "training"
    if as_of <= pd.Timestamp(config.LSTM.val_end):
        return "validation"
    return "test"


def trajectory_table(df: pd.DataFrame, as_of: pd.Timestamp,
                     forecast: dict) -> pd.DataFrame:
    """The 7-day projection beside what the market actually did.

    Realised returns use the same definition and the same denominator as the
    projected ones -- simple return against the origin close -- so the two
    columns are directly comparable and the error column is meaningful.

    Sessions past the end of the data are left blank rather than dropped: the
    horizon is always 7 rows, so a partially-realised forecast is visibly
    partial instead of looking like a shorter forecast.
    """
    future = df[df["date"] > as_of].head(config.LSTM.horizon)
    dates = future["date"].tolist()
    closes = future["close"].astype(float).tolist()
    origin_close = forecast["origin_close"]

    rows = []
    for i in range(config.LSTM.horizon):
        projected = forecast["path_return_pct"][i]
        row = {
            "Session": f"t+{i + 1}",
            "Date": dates[i].date().isoformat() if i < len(dates) else "—",
            "Projected level": f"{forecast['prices'][i]:,.2f}",
            "Realised level": "—",
            "Projected return": f"{projected:+.2f}%",
            "Realised return": "—",
            "Error (pp)": "—",
        }
        if i < len(closes):
            realised = (closes[i] / origin_close - 1) * 100
            row["Realised level"] = f"{closes[i]:,.2f}"
            row["Realised return"] = f"{realised:+.2f}%"
            row["Error (pp)"] = f"{projected - realised:+.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


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

    c1, c2, c3, c4, c5 = st.columns(5, border=True)
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
        st.caption("20-day smoothed sentiment, percentile-ranked against every "
                   "session up to the selected date only — so 50 means a typical "
                   "day so far, and a historical date shows the reading that was "
                   "actually available on it.")

    if forecast:
        st.subheader("Projected trajectory")
        st.dataframe(trajectory_table(df, as_of, forecast),
                     width="stretch", hide_index=True)

        n_realised = int((df["date"] > as_of).sum())
        if n_realised == 0:
            st.caption("No sessions follow the selected date in this dataset, so "
                       "every horizon is still open. Pick an earlier date to see "
                       "the forecast scored against what actually happened.")
        else:
            note = ("Realised columns compare the forecast against the sessions "
                    "that actually followed. Error is projected minus realised, "
                    "in percentage points.")
            if n_realised < config.LSTM.horizon:
                note += (f" Only {n_realised} of {config.LSTM.horizon} sessions "
                         "have happened yet; the rest stay blank.")
            # The forecaster was fitted on train and model-selected on val, so a
            # date from either is not evidence about forecast quality. Saying so
            # here matters more than usual: this table is the most tempting place
            # in the dashboard to read an in-sample fit as a result.
            split = _split_of(as_of)
            if split != "test":
                note += (f" **This date is in the {split} period** — the model saw "
                         "it during fitting, so treat the agreement below as an "
                         "in-sample fit, not as forecast accuracy.")
            st.caption(note)


def tab_committee(as_of: pd.Timestamp) -> None:
    st.caption("Three agents run in sequence via LangGraph. Findings and the "
               "final directive are computed deterministically; the language "
               "model only writes the explanation.")
    if not (require(config.MODELS_OUT / "bilstm.pt", "The investment committee")
            and require(config.CHROMA_DIR, "The investment committee")):
        return
    if not st.button("▶︎ Convene the investment committee", type="primary"):
        st.info("Press the button to run the committee for the selected date.")
        return

    result = run_committee_cached(str(as_of.date()))
    risk = result["risk"]

    # A key that has stopped working degrades to templates silently. Say so
    # here rather than letting the report claim an author it does not have.
    if result.get("narrative_error"):
        st.warning(
            "**The narrative below was written by the template renderer, not by "
            "the language model.** The API call failed, so the prose fell back — "
            "the findings and the directive are unaffected, since those are "
            f"computed in Python either way.\n\n`{result['narrative_error']}`")

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


def tab_evaluation() -> None:
    fm = load_json(config.MODELS_OUT / "forecast_metrics.json")
    if fm:
        st.subheader("Bi-LSTM forecaster — held-out test period")
        t = fm["test"]
        st.caption(f"Test window {fm['split_dates']['test_range'][0]} → "
                   f"{fm['split_dates']['test_range'][1]} · {t['n_samples']} windows, "
                   "never seen during training or model selection.")
        c = st.columns(5, border=True)
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
        style_fig(fig, height=420, legend=LEGEND_BELOW_AXIS,
                  title="Error and directional accuracy by horizon")
        fig.update_xaxes(title_text="Forecast horizon (sessions)", dtick=1)
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
            fig.add_trace(go.Bar(x=sub["mode"], y=sub["ndcg@k"],
                                 name=f"{style} queries",
                                 marker_color=ACCENT if style == "keyword" else UP,
                                 hovertemplate="%{y:.4f}<extra>%{fullData.name}</extra>"))
        style_fig(fig, height=380,
                  title="nDCG@10 by retrieval mode and query style")
        fig.update_layout(barmode="group")
        fig.update_yaxes(title_text="nDCG@10")
        st.plotly_chart(fig, width="stretch")
        st.caption("BM25 wins on keyword queries and collapses on natural-language "
                   "ones; the dense index does the reverse. The fused modes are the "
                   "only ones that stay strong on both — robustness, not a bigger "
                   "peak, is what hybrid retrieval buys.")

    sv = load_json(config.REPORTS / "sentiment_validation.json")
    if sv:
        st.subheader("Sentiment engine validation")
        c = st.columns(4, border=True)
        def _p(block: dict) -> str:
            # Quote the dependence-aware p when the report carries one; older
            # report files only have the i.i.d. Pearson p.
            boot = block.get("p_value_block_bootstrap")
            return (f"block-bootstrap p={boot:.4f}" if boot is not None
                    else f"p={block['p_value']:.1e} (i.i.d.)")

        c[0].metric("Same-day correlation", f"{sv['corr_same_day']['r']:+.3f}",
                    _p(sv["corr_same_day"]))
        c[1].metric("Next-day correlation", f"{sv['corr_next_day']['r']:+.3f}",
                    _p(sv["corr_next_day"]))
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
        c = st.columns(4, border=True)
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
    c = st.columns(4, border=True)
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
    style_fig(fig, height=380, title="Corpus coverage and sentiment by year")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title_text="Headlines", secondary_y=False)
    fig.update_yaxes(title_text="Mean sentiment", secondary_y=True)
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
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("📈 FinAgent-Pulse")
    st.caption("Multi-agent quantitative trading & sentiment analysis using "
               "Hybrid RAG and time-series forecasting — S&P 500, 2018–2024")

    missing = missing_artifacts()
    if any(label == "Processed features" for label, _, _ in missing):
        st.error("**The pipeline has not been run yet.** The dashboard needs the "
                 f"processed feature table at `{config.FEATURES}`.\n\n" + BUILD_HINT)
        st.stop()
    if missing:
        lines = "\n".join(f"- **{label}** — `{path.name}` missing, needed by {needed_by}"
                           for label, path, needed_by in missing)
        st.warning("Some artefacts are missing; the tabs that need them will say so.\n\n"
                   + lines + "\n\n" + BUILD_HINT)

    try:
        df = load_features()
    except FileNotFoundError as exc:
        st.error(f"Could not load the feature table: {exc}\n\n" + BUILD_HINT)
        st.stop()

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

    t1, t2, t3, t4 = st.tabs(
        ["Dashboard", "Investment Committee", "Evaluation", "Corpus"])
    with t1:
        tab_dashboard(df, as_of)
    with t2:
        tab_committee(as_of)
    with t3:
        tab_evaluation()
    with t4:
        tab_corpus(df)


if __name__ == "__main__":
    main()
