"""News memory (LIVE-VIEW + NARRATION ONLY — firewalled from scoring/backtest/panel).

A local, timestamped store of Intrinio headline+summary articles with versioned LLM
extractions, so narration can reference recent AND historically-material events and
cite the article. NOTHING here is ever a feature, scored, or point-in-time-joined into
the panel — timestamps exist only so it COULD be made point-in-time later.

  store    -> NewsStore (news.sqlite): raw articles + versioned extractions + recall
  extract  -> LLM extraction with a number-fabrication guard + heuristic fallback
  curate   -> materiality + source-credibility + dedup ("good article" filter)
  ingest   -> fetch → upsert → extract-missing (quota-capped); watchlist batch
"""

from .curate import credibility, curate, is_good
from .extract import EXTRACT_VERSION, extract_article, heuristic_extraction
from .ingest import (
    backfill_company_news,
    backfill_watchlist,
    ingest_company_news,
    ingest_watchlist,
    watchlist_targets,
)
from .store import NewsStore

__all__ = [
    "NewsStore", "EXTRACT_VERSION", "extract_article", "heuristic_extraction",
    "curate", "credibility", "is_good", "ingest_company_news", "ingest_watchlist",
    "backfill_company_news", "backfill_watchlist", "watchlist_targets",
]
