"""Continuous operation (Phase 5): ingestion jobs, monitoring loop, paper-forward.

Everything here OBSERVES and LOGS; nothing here trains. The frozen artifact is
loaded read-only (stockscan.model.Artifact has no fit method), retraining is a
manual, logged event (scripts/train_model.py + `ops.py paper retrain-record`),
and every scheduled job is idempotent — safe to re-run, logging what it changed.
"""
