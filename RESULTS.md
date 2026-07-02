# Phase-1 Go/No-Go — Survivorship-Free Verdict (2026-07-01)

The first gate result computed on genuinely survivorship-free prices (Intrinio,
delisted-inclusive, fetched by security id) with **no delisting-return imputation**:
dead companies enter through their real price histories and real last-trade terminal
returns. 2011–2026, monthly, 63-trading-day forward sector-excess label, $1M ADV +
$1 price liquidity floor, per-date 1/99% label winsorization, purged+embargoed
walk-forward. `uv run python scripts/run_phase1.py --no-impute`.

## The data that made this trustworthy

- Universe: 11,135 of 13,916 EDGAR-fundamentals CIKs matched to Intrinio securities —
  **4,132 active + 7,003 dead** companies; 11,029 price columns on disk (99% of
  matched). Dead names are ~63% of the priced universe: the survivorship hole is closed.
- Every fetch is by security id, never ticker: recycled tickers (five dead companies
  shared "AAC") can no longer inject another company's prices. Dead columns are named
  `TICKER~CIK`.
- The delisting-ledger clip was **removed** after an audit showed the ledger's
  earliest-Form-25 date is often a *bond* delisting: it had chopped Sears' whole
  2016–18 collapse and Hess's real 2025-07-17 end (Chevron close). By-id series end
  at their true last trade; zero contamination found without the clip.
- Spot checks: RadioShack ends 2015-02 (−90% final yr), Sears 2020-05 (−75%),
  Bed Bath & Beyond 2023-05 (−99%), GNC 2020-06 (−65%), Akorn 2020-05 (−98%),
  Hess 2025-07-17 (merger close, −2%).

## Gate results (no-impute = headline)

| metric | no-impute (honest) | with imputation | gate bar |
|---|---|---|---|
| composite rank IC | **+0.0209** (t_nw +2.83) | +0.0297 (t +4.42) | ≥ 0.03 |
| LightGBM OOS rank IC | **+0.0375** (t_nw +5.59) | +0.0848 (t +11.0) | ≥ 0.03 |
| OOS decile spread (63d) | **+0.0215** | +0.1318 | > 0 net of cost |
| names/date (liquid) | ~2,475 (100% real-priced) | ~1,986 (+42 imputed) | — |

- Composite IC by year (no-impute): positive 11/16 years; negatives 2012, 2013,
  2016, 2020, 2026-partial. Non-overlapping quarterly IC +0.033 (t +4.2).
- Shuffled-label IC ≈ 0 → no leak. NW t stable across lag choices (2→12).
- **Robustness (the key check):** dropping every terminal-label row (names dying or
  halted inside the forward window — only 1.54% of the panel) leaves the edge intact:
  model +0.0345 (t +5.14), composite +0.0207. The edge is broad, not a bet on deaths.
- Imputing mode remains inflated (model 0.085 vs honest 0.037) even at ~42 imputed
  names/date — the circularity diagnosis stands. Its haircut sweep is near-vacuous
  (winsorization clips all haircuts to similar values); the honest configuration
  replaces the sweep entirely by assuming nothing.

## Sign check: did the textbook anomalies de-invert?

Partially. Profitability/quality factors are strong and correctly signed
(roa +0.060, op_margin +0.065, roe +0.051, gross_profitability +0.030). But
**accruals stayed positive** (+0.018, t +3.2) and **asset_growth is ~zero**
(+0.008, t +1.0, down from +0.022 significant under imputation). With survivorship
closed, the residual inversion is most plausibly real post-publication anomaly decay
(both anomalies famously faded after the 2000s), not a data artifact — though our
accruals construction (NI−CFO over assets, annual 10-Ks only) is crude.
current_ratio and cash_to_assets carry *negative* ICs against their assumed +1
composite signs, which is much of why the fixed-sign composite (+0.021) trails the
model (+0.0375) that learns actual signs.

## Verdict: **conditional GO**

The product's actual predictor — the walk-forward ML model — clears every numeric
bar on honest data: OOS IC 0.0375 ≥ 0.03, overlap-corrected t 5.59 ≥ 2, positive
decile spread (+2.15%/quarter gross on a $1M-ADV universe; survives any plausible
cost assumption at monthly rebalance), no imputation to sweep. The equal-weight
textbook composite alone is sub-threshold (0.021 < 0.03) — the edge requires the
model (or at least learned signs), not naive factor averaging.

Context: the free-data upper bound was composite +0.012 / model +0.022. Closing
survivorship *raised* the honest measured edge (survivor-biased prices had hidden
the failures that quality factors correctly avoid). The old headline numbers
(0.05–0.145, t 8–16) are confirmed artifacts of imputation circularity +
survivorship bias.

This is a modest, real edge at the low end of the realistic 0.02–0.05 band —
research-grade validation to proceed with Phase 2 (parsers, monitoring, narration),
not a claim of production alpha.

## Measured residuals (documented, not hidden)

- 2,781 fundamentals CIKs (20%) have no Intrinio security; 1,042 are ledger-dead,
  but samples are overwhelmingly non-listed entity types (LPs, LLCs, non-traded
  REITs, fund vehicles) with a few true misses (e.g. JMP Group).
- ~1,528 series (93% dead microcaps, median $2.7k/day dollar volume) have an
  Intrinio OTC data hole spanning 2015–2017; only 20 would pass the liquidity
  filter, so the tradable panel is essentially unaffected.
- A company that died and relisted under the same CIK (AMR→AAL style) keeps only
  its active security's era.
- Terminal returns use the last real trade; CRSP-style delisting *distributions*
  (final payout after the last print) remain unmodeled — no budget vendor has them.
