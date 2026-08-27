# FinAgent-Pulse

**Multi-Agent Quantitative Trading & Sentiment Analysis Framework using Hybrid RAG and Time-Series Forecasting**

DA592 Term Project — Müge Evran (38033), Gökçe Güleviz (38096), Umut Gümüş (38239)

An end-to-end system that forecasts a 7-day S&P 500 trajectory with a
bidirectional LSTM, reads 12,456 financial headlines with FinBERT, retrieves
context through a hybrid dense + sparse + knowledge-graph RAG stack, and routes
everything through a three-agent investment committee that issues an auditable
Buy/Sell/Hold directive.

📊 **[Full technical report with results and figures →](reports/TECHNICAL_REPORT.md)**

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m finagent_pulse.pipeline
```

The pipeline downloads the data, scores sentiment, builds the indexes, trains
the forecaster, runs every evaluation and writes an executive report. Each stage
is cached, so a rerun only redoes what is missing. First run takes roughly
20–30 minutes on a laptop CPU (FinBERT scoring and index building dominate).

```bash
streamlit run finagent_pulse/app/streamlit_app.py
```

### Running individual stages

```bash
python -m finagent_pulse.pipeline --only forecast     # one stage
python -m finagent_pulse.pipeline --from rag          # resume from a stage
python -m finagent_pulse.pipeline --force             # ignore caches
```

### Optional: LLM-written narratives

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Without a key the system runs completely — a template renderer produces the same
report structure. **The key changes the prose, never the decision:** findings and
directives are computed deterministically in Python so they stay reproducible
and testable. See §2.1 of the technical report.

---

## What the system does

| Layer | Implementation |
|---|---|
| **Data** | Kaggle S&P 500 headlines (2018-01-02 → 2024-03-04) + yfinance `^GSPC` OHLCV |
| **Sentiment** | `ProsusAI/finbert`, softmax reduced to a signed intensity in [−1, 1] |
| **Forecasting** | 2-layer Bi-LSTM, 60-day lookback → 7-day cumulative log-return trajectory, 3-seed ensemble |
| **Retrieval** | ChromaDB (dense) + BM25 (sparse) + 56-node entity knowledge graph, fused with weighted RRF |
| **Agents** | LangGraph: Data Analyst → Sentiment Critic → Risk Manager |
| **Interface** | 5-tab Streamlit dashboard |

## Headline results

| Result | Value |
|---|---|
| Directional accuracy @ 7 days (held-out test) | **73.5%** |
| R² in return space / price space | −0.020 / 0.934 |
| Sentiment vs same-day return | r = **+0.573** (p ≈ 1.8 × 10⁻²¹) |
| Sentiment vs next-day return | r = −0.045 (p = 0.50) — **not predictive** |
| Hybrid RAG vs dense-only (macro nDCG@10) | 0.1738 vs 0.1655 (**+5.0%**) |
| BM25 on natural-language queries | nDCG 0.0136 — a **20× collapse** |
| Committee: abstained / traded | 84% / 16%, **75% hit rate when traded** |

Three findings are worth reading in full in the report, because they are
negative or counter-intuitive and they shaped the design:

- **Headline sentiment describes the session it belongs to and does not forecast
  the next one.** By the time a headline exists, the information is priced. This
  is why the committee treats sentiment as confirmation, not prediction.
- **The 0.93 price-space R² is not an achievement** — any model scores it when
  the target is a price level. The return-space R² near zero is the honest
  number, and directional accuracy is the metric that carries real signal.
- **Hybrid retrieval buys robustness, not a higher peak.** Each single retriever
  wins on the query style that suits it and collapses on the other; the fused
  system is the only one that never collapses.

## Project layout

```
finagent_pulse/
├── config.py               all paths, hyper-parameters, calibrated thresholds
├── pipeline.py             staged end-to-end orchestrator
├── evaluation.py           forecaster ablation, sentiment validation, backtest
├── figures.py              report figures
├── data/
│   ├── ingest.py           Kaggle + yfinance download with offline fallback
│   └── preprocess.py       cleaning, trading-day alignment, feature engineering
├── models/
│   ├── sentiment.py        FinBERT scorer + Fear & Greed index
│   └── forecaster.py       Bi-LSTM, leak-safe splits, inference service
├── rag/
│   ├── entities.py         finance lexicon, knowledge graph, query expansion
│   ├── index.py            ChromaDB + BM25 + graph builders
│   ├── hybrid.py           weighted RRF retriever, 4 modes
│   └── evaluate.py         ablation benchmark + weight calibration
├── agents/
│   ├── llm.py              narrative layer (Claude or deterministic templates)
│   └── committee.py        LangGraph three-agent committee
├── knowledge_base/         investment principles (markdown)
└── app/streamlit_app.py    dashboard
```

Generated artifacts land in `data_raw/`, `data_processed/`, `models_out/`,
`rag_index/` and `reports/`, all git-ignored.

## Dashboard

| Tab | Contents |
|---|---|
| **Dashboard** | Price with forecast trajectory and ±1σ band, sentiment bars, Fear & Greed gauge |
| **Investment Committee** | Runs the three agents live and renders the executive report |
| **Hybrid RAG** | Compare all four retrieval modes on any query — the ablation, interactive |
| **Evaluation** | Every metric, ablation table and chart in the report |
| **Corpus** | Coverage statistics and a per-day headline browser |

## Reproducibility

All randomness is seeded: 42/1337/2024 for the forecaster ensemble, 42 for the
retrieval benchmark, 7 for the disjoint weight-calibration split. Splits are
chronological with an embargo at each boundary; scalers are fitted on training
rows only. Re-running the pipeline reproduces every number in the report.

---

*Academic prototype. Not investment advice.*
