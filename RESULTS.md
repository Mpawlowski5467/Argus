# Phase-2 — Walking Skeleton Verdict (2026-07-01)

End-to-end per-ticker serve path over the frozen model: `scripts/analyze.py TICKER
[--as-of DATE]` runs parse → compute → score → narrate, everything keyed off
`available_date <= as_of`. **The skeleton holds**: all three DESIGN.md §2 invariants
are automated tests (`tests/test_serve.py`; suite now 69 green), and the delisted name
flows through the identical code path with zero special-casing.

## What was built

- **Frozen artifact** (`scripts/train_model.py` → `artifacts/model/`): one LightGBM
  fit on the full honest panel (no-impute, universe ticker map; 450,586 rows, 182
  dates, trained through 2026-03-31). `meta.json` records the feature columns,
  cutoff, and config floors. Serving loads the artifact and can only `score()` —
  there is no retrain path (`Artifact` has no fit method by design).
- **Shared transforms** (`prepare_features` / `pit_snapshot` / `liquidity_mask` /
  `add_sector_ranks` in fundamental_panel.py): the training panel and
  `serve.build_cross_section` run the SAME code, so parity is structural. The parity
  test asserts bit-identical rank vectors (`==`, no tolerance) for every company on
  a shared date, plus end-to-end through `analyze()`.
- **Serve path** (`stockscan/serve.py`): PIT snapshot → liquidity floors → sector
  ranks → frozen-model score → cross-sectional percentile/decile → grounded packet →
  narration (template mode; LLM optional). Honesty flags on every result:
  `liquidity_pass`, staleness, and `in_sample` (as-of inside the training window).
- **Rank-basis fix found by the parity requirement**: the panel used to rank-normalize
  features *after* dropping unlabeled rows — i.e. the rank universe was conditioned on
  the future (whether a name later got a label). Ranks now compute per-date on the full
  known-at-date universe, before the label drop. Re-run gate: model OOS IC **+0.0391**
  (t_nw +5.92), decile spread **+0.0229**, composite +0.0209 — slightly *better* than
  the Phase-1 headline (+0.0375/+5.59/+0.0215) and still passing. Trying to make serve
  and train identical surfaced a real (mild) conditioning bug — the invariant earned
  its keep before Phase 3.

## The 5-ticker run (one command each, identical code path)

| company | as-of | latest 10-K | model signal | notes |
|---|---|---|---|---|
| AAPL (mega-cap) | 2026-07-01 | FY2025, 240d old | 96th pct, decile 10 | |
| JPM (financials) | 2026-07-01 | FY2025, 135d old | 85th pct, decile 9 | Finance-sector ranks (leverage read sector-relative) |
| ETSY (mid-cap) | 2026-07-01 | FY2025, 131d old | 17th pct, decile 2 | negative equity handled (ROE −14.8%, leverage 1.39) |
| RDDT (2024 IPO) | 2026-07-01 & 2025-06-30 | FY2025 / FY2024 | 61st pct / 97th pct | at 2025 as-of only ONE 10-K public: growth features NaN → omitted from packet, median-filled at scoring; no crash |
| **BBBY~886158 (delisted)** | **2022-09-30** | FY2021, 161d old | **3rd pct, decile 1** | seven months pre-bankruptcy, through the identical path; flagged in-sample |

The BBBY row is the gate case: a company that died in 2023 produces a complete,
correctly-dated analysis (ROE −321%, revenue −14.8%, composite 5th pct, model decile 1)
with no if-branch anywhere — a dead name is just a column whose prices stop and whose
filings go stale.

## What broke / what the run surfaced

- **Direction-blind narration** (fixed): the template ranked "strongest/weakest" by raw
  sector percentile, so ETSY's 98th-percentile *leverage* (liabilities > assets) was
  listed as a strength. Strong/weak ordering now flips lower-is-better signals and
  annotates them ("98th pct; lower is better"); the LLM system prompt got the same rule.
- **Grounding guard was porous** (found by an adversarial review workflow, fixed +
  regression-tested): (1) the 0.5% *relative* tolerance meant large packet numbers
  blessed nearby fabrications — cik 886158 accepted any figure within ±4,431, and
  fiscal_year 2024 accepted any year 2014–2034; (2) the signed-number regex decomposed
  packet dates like `2026-03-31` into {2026, −3, −31}, whitelisting fabricated negative
  percentages while *falsely rejecting* a legitimately reformatted "March 31, 2026";
  (3) plural "10-Ks" leaked a numeral 10. Now: integers must match exactly, floats get
  a small absolute tolerance only, dates decompose into positive (y, m, d) components.
- Smaller review catches (fixed): `in_sample` flag now accounts for the label horizon
  (training labels at the cutoff are realized over the next 63 trading days, so the
  information window extends past `trained_through`); packet YoY now pairs against the
  latest *earlier-period* filing rather than positionally (a delinquent re-filing could
  pair with itself → fabricated 0.0 delta); `load_artifact` raises a guidance-bearing
  FileNotFoundError and warns on a lightgbm version mismatch; `train_model --eval`
  no longer crashes on panels too short to walk-forward.
- Scoring an historical as-of with the 2026-trained artifact is knowingly in-sample —
  reported honestly via the `in_sample` flag rather than hidden. Walk-forward artifact
  vintages are Phase-3 backtester work, not a serve-path concern.
- Serve loads the full price matrix (~11k columns) per process (~2 min); fine for a
  walking skeleton, needs a cached store before the monitoring loop (Phase 5).

Known accepted residuals (documented, deferred): the liquidity floors run on
*adjusted* closes, so large later splits/dividends can shift a name's historical
universe membership (train and serve identically — parity unaffected; fix = store
unadjusted closes, Phase 3); an illiquid target injected via `include_cik` perturbs
peer ranks by one row (flagged `liquidity_pass=False`; there is no training row to be
parity with in that case); the artifact pins no content hash of the data stores, so
refreshing parquet between train and serve silently changes in-sample cross-sections;
`pit_snapshot` keys "latest" on filing availability, so a delinquent filer re-filing
an old fiscal year after a newer one would briefly surface the older period (rare,
identical in both paths).

## Deferred

- **Phase 3 (backtester/signal mechanics):** hysteresis signals, costs, capacity,
  artifact vintages for honest historical scoring, CPCV/PBO.
- **Phase 4 (narration hardening):** LLM serving via MLX/llama.cpp, SHAP top-k in the
  packet, materiality-gated invocation, faithfulness eval. Template mode is the
  Phase-2 configuration and passes grounding by construction.
- Value features (need PIT market cap), quarterly (10-Q) cadence, distress head.

---

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
