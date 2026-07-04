# ML risk layer — confidence score + large-drawdown head

Status: **spec + in-progress** (decided 2026-07-04). Two new, firewalled, display/risk-only
outputs that sit beside the return signal — never inside it. Same discipline as the
[distress head](../src/stockscan/distress.py): they read the frozen score, never change it,
never touch paper/trade, degrade to `None` when their artifact is absent.

## Why (the reframe)

The return head predicts ONE thing: cross-sectional relative return, ~1 month out. Every
attempt to predict *that* better has died under CPCV (momentum, reversal, low-vol, value).
So we stop adding return-factors and instead add **new, more-learnable targets** — the way
the distress head did (it passed its ranking gate where the factors failed). Two, in order:

1. **Confidence score** — how much to trust a given BUY/HOLD/AVOID call. Built first.
2. **Large-drawdown head** — P(a name craters over the next H months). Built second.

Neither needs new data: fundamentals (EDGAR) + prices (Intrinio) already on disk. (The
individual Intrinio plan has no options/ETF feeds — irrelevant to both heads.)

---

## 1 · Confidence score (DERIVED, not trained)

The trap is inventing a number. So confidence is **derived transparently from the model's
real out-of-sample track record**, not a second model that could overfit or fabricate.

### The anchor — a calibration artifact
`scripts/build_confidence_calibration.py` rebuilds the honest panel exactly as
`train_model.py` does (`build_fundamental_panel`, no-impute, liquidity floors, 1/99 winsor),
runs the purged walk-forward (`model.walk_forward_predict`), buckets the pooled OOS rows into
prediction deciles per date, and records **per decile**:

- `hit_rate` = P(`label_excess > 0`) — how often that decile actually beat the cross-section OOS
- `mean_excess`, `n`, and a Wilson 95% CI on the hit-rate

Frozen to `artifacts/confidence_cal/calibration.json` alongside metadata (trained_through,
n_dates, label horizon, method). Refreshed only when the return model is refrozen — it must
describe the SAME model the serve path scores with.

### The 0–100 number (`src/stockscan/confidence.py`)
Anchored on the decile's **directional edge**, then bounded-adjusted:

```
edge            = |hit_rate(decile) - 0.5|        # strength of the model's view at this decile
conviction_base = clip(edge / EDGE_FULL, 0, 1)    # EDGE_FULL = 0.10  → a 60/40 decile = full base
score = 100 * conviction_base * margin * data_quality * coherence   # each modifier in [~0.7, 1.0]
score = min(score, CEILING)                        # CEILING = 85 — never imply certainty
```

- **margin** — how deep the name sits in its zone (percentile distance past the BUY 80th / AVOID 40th line). Mild (±~15%).
- **data_quality** — stale filing, below the liquidity floor, or `in_sample` each knock it down (flags already in the serve packet).
- **coherence** — `|Σ contrib| / Σ|contrib|` over the SHAP drivers: 1 when they all point one way, ~0 when they cancel. Mild (±~20%).

**Honesty guards (non-negotiable):** the edge here is small (IC ~0.03–0.05), so
`CEILING=85` and `EDGE_FULL=0.10` keep the number from ever implying certainty; and we
**always** surface the raw `hit_rate` + `n` beside the number so it can't decouple from the
track record. HOLD/mid-deciles get low confidence *by construction* (edge ≈ 0) — that's the
honest "the model has no strong view here."

### Surface
`BUY · 71/100 · names like this beat the market 58% of the time OOS (n=340)`.
Attached in `serve.analyze()` as a firewalled `confidence` block AFTER the model block is
fixed (mirrors the `distress` block); display-only, never a feature/trade input.

### Chosen defaults
- Anchor on the existing frozen-vintage OOS folds (recomputed by the builder against the
  same panel recipe) — consistent with paper, no fresh CPCV needed for v1.
- Display = **0–100 score + the hit-rate it's built from** (user's pick).
- v2 (later): fold distress + drawdown odds in as extra downward modifiers; optionally a
  learned quantile-uncertainty head if the derived version proves useful.

---

## 2 · Large-drawdown head (a distress clone)

`src/stockscan/drawdown.py` + `scripts/run_drawdown_head.py` + `artifacts/drawdown_model/`,
mirroring `distress.py` almost line-for-line — same RANK_COLS features, same serve-parity
seam, same purged WF + CPCV, same rare-event scorecard (AUC / precision@decile / calibration).
Only the LABEL changes.

- **Label:** at each monthly rebalance date `d`, `y = 1` if the name's path suffers a
  peak-to-trough drawdown ≤ **X = −30%** over the next **H = 6 months** (path-based, so a
  crash-then-partial-recover still counts); `y = 0` if it survives the window without one;
  row dropped where the forward window runs past the price censor (no fabricated negatives).
- **Terminal handling (open):** default **include delisting-to-zero as a max drawdown**, but
  **report the orthogonality vs the distress score** — if the drawdown head is just distress
  in a hat (high rank correlation, no incremental AUC on non-terminal names), we don't ship it.
- **Gate:** the distress gate (CPCV mean-AUC, precision@decile lift, calibration MAE) **plus**
  the orthogonality check. **Expectation set by distress:** it may pass ranking yet the trade
  overlay says *don't trade it* → it lives as a display/risk flag + a confidence modifier, not
  a return signal.
- **Tunable:** H ∈ {6, 12} months, X ∈ {−30%, −40%}; the builder reports sensitivity.

---

## Firewall (applies to both)

Cost basis, confidence, distress, drawdown — all live-view / display-only. They read the
frozen return score and never write back into score / percentile / decile / drivers / packet /
paper / trade rule. Enforced the distress way: attached after scoring in `serve.analyze`,
optional-load (`None` when unfrozen), and a byte-identical firewall test (the return result is
unchanged whether or not the head is present).

## Build order
Confidence first (derived from what already exists → fast, upgrades every call + the holdings
panel today) → drawdown head as a proper experiment → then pipe drawdown odds back into
confidence. Feeds the holdings/position panel (see the web-personalization work).
