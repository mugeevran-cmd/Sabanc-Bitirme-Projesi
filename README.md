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

## Setup for the team

Three steps, on any of the team's machines. **Requires Python 3.10 or newer.**

### macOS / Linux

```bash
git clone https://github.com/mugeevran-cmd/Sabanc-Bitirme-Projesi.git
cd Sabanc-Bitirme-Projesi
./setup.sh
```

Then start the dashboard:

```bash
./.venv/bin/streamlit run finagent_pulse/app/streamlit_app.py
```

### Windows (PowerShell)

```powershell
git clone https://github.com/mugeevran-cmd/Sabanc-Bitirme-Projesi.git
cd Sabanc-Bitirme-Projesi
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
Invoke-WebRequest -Uri "https://github.com/mugeevran-cmd/Sabanc-Bitirme-Projesi/releases/download/artifacts-v1/artifacts.zip" -OutFile artifacts.zip
Expand-Archive artifacts.zip -DestinationPath . -Force
.venv\Scripts\streamlit run finagent_pulse\app\streamlit_app.py
```

### How long this takes

`setup.sh` installs the dependencies and then downloads **`artifacts.zip`**, a
46 MB snapshot of the generated dataset, model checkpoint and search indexes,
published as a [release asset](https://github.com/mugeevran-cmd/Sabanc-Bitirme-Projesi/releases/tag/artifacts-v1)
rather than committed, so a clone stays small. With it the pipeline is skipped
entirely and setup finishes in about **five minutes** — essentially the time it
takes pip to download PyTorch.

Nothing in that archive is precious: every byte is reproducible from seeds. To
rebuild from source instead, delete the four generated folders and re-run:

```bash
rm -rf data_raw data_processed models_out rag_index
python -m finagent_pulse.pipeline
```

That path downloads the Kaggle corpus, pulls ~500 MB of FinBERT and MiniLM
weights, scores 12,456 headlines, builds three search indexes and trains a
3-seed LSTM ensemble. Measured end-to-end on an M2 MacBook from a clean clone:
**8 minutes** (sentiment 144 s, evaluation 221 s, forecaster 68 s, indexes 27 s).
Stages are cached individually, so it is resumable.

That clean-clone run reproduced the reported metrics **bit-for-bit** — RMSE
0.014947, R² −0.020412, directional accuracy 73.5426%, skill +2.973156% — which
is what the fixed seeds are there to guarantee.

The dashboard refuses to start before either path has completed, and says so
explicitly rather than failing obscurely.

---

## Running individual stages

```bash
python -m finagent_pulse.pipeline --only forecast     # one stage
python -m finagent_pulse.pipeline --from rag          # resume from a stage
python -m finagent_pulse.pipeline --force             # ignore caches
```

Stages, in order: `ingest → sentiment → features → rag → forecast → evaluate → report`

### Tests

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/pytest
```

114 tests, ~10 seconds, no pipeline run required — the retriever and the language
model are stubbed and the price/sentiment frames are synthetic. They pin the
things that would otherwise fail silently: the Risk Manager's decision table,
the leak-safety of the windowing and the train/val/test embargo, the
point-in-time Fear & Greed definition, the two non-standard retrieval metrics,
the autocorrelation-aware significance test, the BM25 tokeniser on both the
index and the query side, the cross-checks each agent narrates, and that the
committed reports in `reports/` still carry the fields the code that writes
them produces.

The last two are there because both failed silently once: a punctuation-splitting
tokeniser made BM25 look 4× worse than it is on natural-language queries
(§5.1.1), and the evaluate stage went two commits without being re-run while the
write-up quoted a p-value that existed nowhere in the repository.

### Optional: LLM-written narratives

The three committee agents can have their prose written by Claude instead of by
the template renderer. The findings handed to the model already carry all three
channels: the computed numbers, the headlines hybrid retrieval surfaced for that
date, and the investment principles the Risk Manager consulted.

Set the key in your shell profile, never in a file inside this repository:

```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-your-key-here' >> ~/.zshrc && exec zsh
```

Check it took, without printing the key:

```bash
[ -n "$ANTHROPIC_API_KEY" ] && echo "key is set (${#ANTHROPIC_API_KEY} chars)"
```

**This repository is public.** A key committed here is a key someone else
spends, and deleting the commit does not un-publish it. Two guards are in place:
`.gitignore` covers `.env*` and `secrets.*`, and a pre-commit hook refuses any
commit containing something shaped like a live key. Enable the hook once per
clone:

```bash
git config core.hooksPath .githooks
```

If a key is ever exposed, revoke it at https://console.anthropic.com rather than
trying to rewrite history. Share keys with teammates through a password manager,
not through chat.

Roughly $0.03 per committee run at `claude-sonnet-5` rates: about 2,600 input
and 2,700 output tokens across the three agents. The dashboard caches a run per
date, and the backtest passes `narrative=False`, so neither repeats the calls.

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
| **Interface** | 4-tab Streamlit dashboard |

## Headline results

| Result | Value |
|---|---|
| Directional accuracy @ 7 days (held-out test) | **73.5%** (price-only: 67.7%) |
| Skill vs naive persistence (RMSE) | +2.97% — **below** price-only's +3.77% |
| R² in return space / price space | −0.020 / 0.934 |
| Sentiment vs same-day return | r = **+0.573** (block-bootstrap p = 0.0002) |
| Sentiment vs next-day return | r = −0.045 (p = 0.50) — **not predictive** |
| `hybrid_kg` vs dense-only (macro nDCG@10) | 0.2063 vs 0.1738 (+18.7%, **p < 0.001**) |
| `hybrid` vs sparse-only (macro nDCG@10) | 0.1984 vs 0.1922 (+3.2%, p = 0.43 — **not a difference**) |
| BM25 on natural-language queries | nDCG 0.0345 — **1.9× below** the dense index |
| Committee: abstained / traded | 84% / 16%; 75% hit rate on **12 trades** (9/12, p = 0.07, 95% CI [0.47, 0.91]) |

Five findings are worth reading in full in the report, because they are
negative or counter-intuitive and they shaped the design:

- **Headline sentiment describes the session it belongs to and does not forecast
  the next one.** By the time a headline exists, the information is priced. This
  is why the committee treats sentiment as confirmation, not prediction.
- **The 0.93 price-space R² is not an achievement** — any model scores it when
  the target is a price level. The return-space R² near zero is the honest
  number, and directional accuracy is the metric that carries real signal.
- **Sentiment buys direction and costs magnitude.** It lifts directional
  accuracy 67.7% → 73.5%, but the price-only model has the better RMSE skill
  against a naive baseline (+3.77% vs +2.97%). The decision layer consumes
  direction and ignores magnitude, so the combined model is the right pick —
  but both numbers belong in the same sentence. See §4.3.
- **Hybrid retrieval beats the dense index, and does not beat BM25.** Against
  dense-only the gain is large and unambiguous (+18.7% macro nDCG, p < 0.001).
  Against BM25 it is nothing pooled (+3.2%, p = 0.43) — fusion *loses* on keyword
  queries (−0.024 nDCG, p = 0.019) and *wins* on natural-language ones (+0.037,
  p = 0.002), and the two cancel. Robustness across query styles is what fusion
  buys; a higher peak is not. See §5.4.1.
- **The previous version of the two rows above was wrong, and a tokenisation bug
  is why.** The BM25 index and the query path both used `str.split()`, so
  punctuation stayed attached to the term and every natural-language query lost
  one of its two entity names to a trailing question mark. That understated BM25
  by roughly 4× on semantic queries and turned "fusion beats sparse-only" into a
  headline result it is not. See §5.1.1 — it is the most useful thing in this
  report to read before trusting a retrieval benchmark.

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
| **Evaluation** | Every metric, ablation table and chart in the report |
| **Corpus** | Coverage statistics and a per-day headline browser |

## Reproducibility

All randomness is seeded: 42/1337/2024 for the forecaster ensemble, 42 for the
retrieval benchmark, 7 for the disjoint weight-calibration split. Splits are
chronological with an embargo at each boundary; scalers are fitted on training
rows only. Re-running the pipeline reproduces every number in the report.

## License

None. This is coursework for DA592, published so it can be read and reviewed —
**all rights reserved by the authors**. Nothing here is licensed for reuse,
redistribution or derivative work; if you want to use any part of it, ask us.

---

*Academic prototype. Not investment advice.*
