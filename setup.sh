#!/usr/bin/env bash
# FinAgent-Pulse — one-command setup.
#
#   ./setup.sh          full setup: venv, dependencies, then the whole pipeline
#   ./setup.sh --deps   dependencies only, skip the pipeline
#
# The pipeline is cached stage by stage, so re-running this is cheap.
set -euo pipefail

cd "$(dirname "$0")"

PY_MIN="3.10"
PYTHON="${PYTHON:-python3}"

echo "──────────────────────────────────────────────────────────"
echo " FinAgent-Pulse setup"
echo "──────────────────────────────────────────────────────────"

# ---- 1. Python version check -------------------------------------------
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "✗ '$PYTHON' not found. Install Python $PY_MIN or newer, then retry."
    echo "  (or point this script at a specific interpreter: PYTHON=python3.12 ./setup.sh)"
    exit 1
fi

PY_VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "✗ Python $PY_VER found, but $PY_MIN or newer is required."
    exit 1
fi
echo "✓ Python $PY_VER"

# ---- 2. Virtual environment --------------------------------------------
if [ ! -d .venv ]; then
    echo "→ Creating virtual environment (.venv)"
    "$PYTHON" -m venv .venv
else
    echo "✓ Virtual environment already exists"
fi

# ---- 3. Dependencies ----------------------------------------------------
echo "→ Installing dependencies (a few minutes; PyTorch is a large download)"
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"

if [ "${1:-}" = "--deps" ]; then
    echo
    echo "Dependencies only. Run the pipeline yourself with:"
    echo "  ./.venv/bin/python -m finagent_pulse.pipeline"
    exit 0
fi

# ---- 4. Prebuilt artifacts, if the repo ships them ----------------------
# artifacts.zip is a convenience snapshot of the generated data, model
# checkpoint and search indexes. Unpacking it skips the pipeline entirely.
# Everything in it is reproducible -- delete it and run the pipeline to
# regenerate from scratch.
if [ -f artifacts.zip ] && [ ! -d rag_index ]; then
    echo "→ Found artifacts.zip — unpacking prebuilt data and indexes"
    if unzip -oq artifacts.zip; then
        echo "✓ Artifacts unpacked; skipping the ~8 minute pipeline"
        echo "  (delete data_raw/ data_processed/ models_out/ rag_index/ and"
        echo "   re-run this script to rebuild everything from source)"
        cat <<'DONE'

──────────────────────────────────────────────────────────
 Setup complete. Start the dashboard with:

   ./.venv/bin/streamlit run finagent_pulse/app/streamlit_app.py
──────────────────────────────────────────────────────────
DONE
        exit 0
    fi
    echo "! Could not unpack artifacts.zip; falling back to the full pipeline"
fi

# ---- 5. Pipeline --------------------------------------------------------
echo
echo "→ Running the pipeline. Takes about 8 minutes on a modern laptop:"
echo "    · downloads the Kaggle headline corpus and yfinance prices"
echo "    · downloads FinBERT and MiniLM (~500 MB of model weights)"
echo "    · scores 12,456 headlines, builds the indexes, trains the forecaster"
echo "    · runs every evaluation and writes the reports"
echo "  Each stage is cached, so re-running only redoes what is missing."
echo
./.venv/bin/python -m finagent_pulse.pipeline

# ---- 6. Done ------------------------------------------------------------
cat <<'DONE'

──────────────────────────────────────────────────────────
 Setup complete. Start the dashboard with:

   ./.venv/bin/streamlit run finagent_pulse/app/streamlit_app.py

 It opens at http://localhost:8501
──────────────────────────────────────────────────────────
DONE
