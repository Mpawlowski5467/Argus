#!/usr/bin/env bash
# Rebuild the throwaway Argus instance the REA Phase-1 baseline was captured against.
#
#   bash demo_setup.sh <demo-dir>          # builds <demo-dir>/{artifacts,data,demo.env.sh}
#   source <demo-dir>/demo.env.sh && uv run --extra web python scripts/argus_web.py   # :8024
#   python3 -I timing_proxy.py 8025 8024 <demo-dir>/timing.jsonl                      # :8025
#   rea capture-browser-scenario scenario.json --format json --full-output > result.json
#
# Isolation: every MUTABLE store (ops_state with the demo book, narration/panel caches,
# locks, backups, logs, matrix cache) lives under <demo-dir>. Real inputs are reached
# read-only: model heads + paper store are COPIED, public-data caches are sqlite-.backup
# COPIES, price/fundamental parquet is SYMLINKED (never written by the web app). The
# Intrinio key from .env is pre-empted with "" (config._load_dotenv uses setdefault), so
# the demo never spends the paid quota — live quotes are null and profiles/caps come from
# the copied caches only. Run from the repo root.
set -euo pipefail
DEMO="${1:?usage: demo_setup.sh <demo-dir>}"
REPO="$(pwd)"
A="$DEMO/artifacts"; D="$DEMO/data"
rm -rf "$A" "$D"; mkdir -p "$A/backups" "$A/locks" "$D/parquet/matrix_cache" "$D/raw" "$D/logs"
for d in model confidence_cal distress_model drawdown_model paper_forward; do cp -R "$REPO/artifacts/$d" "$A/$d"; done
for db in themes marketcap profiles news; do sqlite3 "$REPO/artifacts/$db.sqlite" ".backup '$A/$db.sqlite'"; done
for p in prices fundamentals fundamentals_wide.parquet intrinio_universe.parquet delistings.parquet; do
  ln -s "$REPO/data/parquet/$p" "$D/parquet/$p"; done
cp "$REPO/data/raw/company_tickers.json" "$D/raw/"
cat > "$DEMO/demo.env.sh" <<EOF
export STOCKSCAN_ARTIFACTS_DIR="$A"
export STOCKSCAN_DATA_DIR="$D"
export STOCKSCAN_OPS_STATE_PATH="$A/ops_state.sqlite"
export STOCKSCAN_BACKUPS_DIR="$A/backups"
export STOCKSCAN_INTRINIO_KEY=""
export STOCKSCAN_PRICE_PROVIDER="yfinance"
export STOCKSCAN_TIINGO_TOKEN=""
export STOCKSCAN_WEB_HOST="127.0.0.1"
export STOCKSCAN_WEB_PORT="8024"
export STOCKSCAN_NOTIFY="off"
EOF
# shellcheck disable=SC1091
source "$DEMO/demo.env.sh"
# the demo book: scripts/capture_screenshots.py's seed (5 watched names, 3 invented positions)
uv run --extra web python -I -c "
import importlib.util, os, sys
from pathlib import Path
sys.path.insert(0, 'src')
spec = importlib.util.spec_from_file_location('cap', 'scripts/capture_screenshots.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.seed_demo(Path(os.environ['STOCKSCAN_OPS_STATE_PATH']))"
echo "demo ready -> $DEMO (source $DEMO/demo.env.sh)"
