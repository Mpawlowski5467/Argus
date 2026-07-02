"""Central configuration and the locked project decisions.

Everything a reader needs to know about "what did we decide" lives here or in
DESIGN.md §10. Values can be overridden via environment variables so the code
stays deterministic while remaining tweakable for experiments.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- paths --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("STOCKSCAN_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"          # untouched downloads (FSDS zips, Stooq dumps)
PARQUET_DIR = DATA_DIR / "parquet"  # the immutable point-in-time panel + prices
ARTIFACTS_DIR = Path(os.environ.get("STOCKSCAN_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
DUCKDB_PATH = DATA_DIR / "stockscan.duckdb"

for _d in (RAW_DIR, PARQUET_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- EDGAR --------------------------------------------------------------------
# SEC requires a descriptive User-Agent that includes a contact address, and
# enforces a hard 10 req/s per IP. We stay under it.
EDGAR_USER_AGENT = os.environ.get(
    "STOCKSCAN_EDGAR_UA", "stock-analysis research mpawlowski5467@gmail.com"
)
EDGAR_MAX_RPS = float(os.environ.get("STOCKSCAN_EDGAR_MAX_RPS", "8"))

# --- price provider -----------------------------------------------------------
# "yfinance" (free, survivorship-biased) or "tiingo" (paid, delisted-inclusive).
PRICE_PROVIDER = os.environ.get("STOCKSCAN_PRICE_PROVIDER", "yfinance")
TIINGO_TOKEN = os.environ.get("STOCKSCAN_TIINGO_TOKEN", "")

# --- local LLM (NARRATE stage) ------------------------------------------------
# OpenAI-compatible endpoint: Ollama (http://localhost:11434/v1) or llama.cpp/MLX server.
LLM_BASE_URL = os.environ.get("STOCKSCAN_LLM_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("STOCKSCAN_LLM_MODEL", "qwen2.5:32b")

# --- locked modeling decisions (DESIGN.md §10) --------------------------------
LABEL_HORIZON_DAYS = 63           # forward return horizon (~3 months)
AVAILABILITY_LAG_BDAYS = 1        # a filing's numbers are usable at filed + 1 business day
MIN_SECTOR_BUCKET = 20            # min names per (date x sector) before broad fin/non-fin fallback
FEATURE_COVERAGE_FLOOR = 0.70     # drop / bucket-fallback any feature below this per-date coverage

# tradable universe floors
MIN_MARKET_CAP = 100_000_000      # $100M
MIN_DOLLAR_VOLUME = 1_000_000     # $1M 20-day median dollar volume

# delisting-return convention — labeled ESTIMATES; the sweep is a Phase-1 gate
DELISTING_RETURN = {
    "distress": -0.70,
    "going_dark": -1.00,
    "mna": None,                  # carry last traded / deal price
}
DELISTING_HAIRCUT_SWEEP = (-0.30, -0.55, -0.70, -1.00)

# --- go/no-go gate (Phase 1) --------------------------------------------------
GATE_MIN_IC = 0.03                # out-of-sample mean rank IC
GATE_MIN_IC_TSTAT = 2.0           # overlap-corrected t-stat
