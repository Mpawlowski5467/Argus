# Reversal / Low-Vol / Illiquidity Test — All Three Fail; st_rev's Marginal Bar-Pass Dies at the Matched Horizon (2026-07-03)

Direct follow-up to the momentum test (real standalone signal but **non-additive** — did
not beat the fundamentals-only model under walk-forward + CPCV, so it was NOT promoted;
see the `momentum-non-additive` memory / `feature/momentum-factor`). Same honest harness,
three new Tier-0 **price** features attached to the SAME 450k-row panel (they never filter
rows, so every arm scores the identical cross-section — only the feature set changes):
short-term reversal (`st_rev`, raw trailing 21d return), low-vol (`low_vol`, −126d realized
vol), Amihud illiquidity (`amihud`, trailing |ret|/dollar-volume).

**Verdict: do NOT promote any of them.** `low_vol`, `amihud`, and `+all3` fail outright at the
model level. `st_rev` nominally meets the letter of the bar at the shipped 63d horizon (beats
baseline on WF IC/t AND CPCV mean at every seed, tail intact) and its 63d walk-forward lift is
even block-bootstrap-significant — **but the follow-up matched-horizon test (Path 2, below) kills
it**: at the reversal's native ~1-month horizon st_rev is dead (WF Δ +0.0007, bootstrap CI crosses
zero, CPCV Δ ≈ 0). The 63d bump is horizon-fragile, interaction-driven, and one-of-five features
tried — an artifact, not a reversal edge. Per protocol I STOPPED before touching serve/retraining.
The shipped model, serve path, and paper-forward book are byte-identical (`price_features=False`).

`uv run python scripts/run_reversal_test.py` (no-impute, 450,586 rows, 182 dates, ~2,475
names/date). rank_ic is SIGNED, so standalone ICs are reported with their real sign.

## The three candidates on the DATA bars

| feature | coverage | orthogonality (mean \|corr\| to fundamentals) | standalone signed IC (t_nw) |
|---|---|---|---|
| `st_rev`  | 100% | **0.014** (orthogonal) | **+0.0060 (t +0.95)** — expected −; the 21d reversal is *washed out* at the 63d label |
| `low_vol` | 85%  | 0.233 (overlaps quality) | +0.0665 (t +3.02) |
| `amihud`  | 100% | 0.181 (overlaps size/quality) | −0.0615 (t −5.22) — sign flips vs textbook inside a liquidity-floored universe |

The honest caveat held: short-term reversal is a ~1-month effect, and against the 63-day
label its standalone IC is ~0 and **insignificant**. low_vol and amihud have strong (signed)
standalone ICs but are *not* orthogonal — they lean on the same quality/size axis the
fundamentals already price.

## …and at the MODEL level (walk-forward + CPCV — the bar that counts)

| feature set | WF IC | t_nw | decile | CPCV mean | CPCV 5th-pct | frac>0 |
|---|---|---|---|---|---|---|
| **baseline (10 fundamentals)** | **+0.0391** | **+5.92** | +0.0229 | **+0.0369** | +0.0134 | 100% |
| + st_rev  | +0.0419 | +6.38 | +0.0246 | +0.0390 | +0.0147 | 100% |
| + low_vol | +0.0414 | +5.31 | +0.0258 | **+0.0238** | **−0.0279** | **84%** |
| + amihud  | +0.0418 | +6.31 | +0.0261 | +0.0372 | +0.0107 | 100% |
| + all3    | +0.0421 | +5.68 | +0.0276 | **+0.0251** | **−0.0255** | **87%** |

- **low_vol** and **+all3** COLLAPSE under CPCV — mean craters to ~+0.024, the 5th-pct goes
  **negative**, frac>0 drops to ~85%. A walk-forward "win" that dies under CPCV: exactly the
  failure mode the bar exists to catch. Do NOT promote.
- **amihud** looks good on walk-forward but its CPCV mean barely moves (+0.0372 vs +0.0369,
  Δ ≈ 0) and its 5th-pct is *worse* than baseline (+0.0107 vs +0.0134). Paired per-combo it is
  a wash/negative (Δmean −0.0009, only 51% of combos improve, paired t −1.17). Fails the
  CPCV-mean half of the bar. Do NOT promote.
- **st_rev** beats baseline on WF IC/t AND CPCV mean — the only feature that does.

## st_rev robustness — the one that clears the bar (thin, but seed-robust)

Because meeting the bar means STOP-and-report, a +0.002 CPCV delta was not taken at face
value. Two checks (LightGBM bagging is seeded, and CPCV combos are *paired*):

| seed | WF Δ (st_rev − base) | CPCV mean Δ |
|---|---|---|
| default | +0.0028 | +0.0021 |
| 0   | +0.0045 | +0.0015 |
| 7   | +0.0048 | +0.0006 |
| 42  | (paired, below) | +0.0011 |
| 123 | +0.0029 | +0.0019 |

- **Seed-robust and always positive:** WF Δ ∈ [+0.003, +0.005] and CPCV mean Δ ∈ [+0.0006,
  +0.0021] — positive at **5/5 seeds**, never flat or negative. The baseline CPCV mean itself
  wobbles ±~0.0004 across seeds, so most of the st_rev lift sits above the seed-noise floor.
- **But not uniform and interaction-driven:** the paired per-combo CPCV delta (seed 42) is
  +0.0011 with only **67% of the 45 combos improving** (a third get *worse*); the paired
  t = +3.00 is inflated because CPCV combos share overlapping training data (effective N ≪ 45).
  And st_rev's own standalone IC is insignificant — the model-level gain comes from
  *interactions/conditioning*, not a direct signal, which is more fragile.
- The WF lift (~+0.004) is consistently **larger** than the CPCV lift (~+0.0014), hinting the
  reversal edge is period-dependent (WF's later expanding folds weight recent regimes more).

## Path 2 — matched-horizon + block-bootstrap (in-sample, since paper-forward can't resolve a ~0.001 effect for years)

The paper-forward OOS window can't confirm a +0.0014 CPCV effect against ~0.10 monthly-IC
noise for years, so the verdict was tightened in-sample instead (`scripts/run_reversal_matched_horizon.py`):
(1) rebuild the panel with a **21-day** label — the reversal's native horizon — and re-run
everything; (2) a construction scan of trailing windows {5,10,21,42,63}; (3) a
**moving-block bootstrap** of the seed-averaged per-date ΔIC (the honest significance test the
paired CPCV t=3.00 wasn't — CPCV combos share training data, so that t is inflated).

**Construction scan — signed cross-sectional IC of the raw trailing-window return** (reversal ⇒
NEGATIVE; t_nw in parens):

| label | w5 | w10 | w21 | w42 | w63 |
|---|---|---|---|---|---|
| 21d | −0.0042 (−0.6) | −0.0018 (−0.2) | +0.0013 (+0.2) | +0.0018 (+0.2) | +0.0075 (+0.8) |
| 63d | +0.0068 (+1.3) | +0.0083 (+1.4) | +0.0105 (+1.5) | +0.0158 (+1.6) | +0.0196 (+1.8) |

Reversal (negative IC) shows up ONLY at w5/w10 vs the 21d label and is **insignificant**;
everything else is positive (weak *continuation*). **There is no meaningful short-term reversal
in this liquid, survivorship-free universe** — the raw trailing return behaves like weak
momentum, strongest at longer windows.

**Model-level, matched vs shipped horizon** (seed-avg WF; block-bootstrap of per-date ΔIC):

| label | WF baseline IC (t) | WF +st_rev IC (t) | WF ΔIC | bootstrap ΔIC 95% CI | P(ΔIC>0) | CPCV Δmean (frac>0) |
|---|---|---|---|---|---|---|
| **21d** (native) | +0.0250 (4.87) | +0.0257 (4.72) | +0.0007 | **[−0.0042, +0.0057]** | 62% | −0.0001 (47%) |
| **63d** (shipped) | +0.0391 (5.89) | +0.0432 (6.30) | +0.0041 | [+0.0021, +0.0062] | 100% | +0.0011 (67%) |

At the **21-day horizon where reversal should be strongest, st_rev is dead**: WF Δ +0.0007 with
a bootstrap CI straddling zero, and CPCV Δ −0.0001 — the t-stat even *drops*. The 63d benefit
does NOT replicate at the matched horizon; it is **horizon-fragile**. So whatever st_rev adds at
63d is weak short-horizon continuation working through interactions, not a reversal signal.

## Verdict & recommendation — do NOT promote

`st_rev` nominally passes the mechanical bar at 63d (WF ↑, CPCV mean ↑), and its 63d walk-forward
ΔIC (+0.0041) is even block-bootstrap-significant. **But Path 2 is decisive against promotion:**

1. **The reversal thesis is falsified.** At the native 21-day horizon st_rev is dead (bootstrap CI
   crosses zero, CPCV Δ ≈ 0). A real ~1-month effect would be *strongest* at ~1-month labels; this
   is *absent* there and only appears at 63d — the fingerprint of a horizon-specific artifact, not
   a stable economic signal.
2. **The one positive is the weakest kind of evidence the project trusts.** The 63d WF significance
   comes from a single train/test scheme; CPCV (the stricter lens the bar was built around) says
   +0.0011 marginal, 67% of combos. And st_rev is **one of five price features tried** (mom_12_1,
   mom_6_1, st_rev, low_vol, amihud) — finding one WF-significant improvement across five candidates
   is textbook multiple-comparisons.
3. **Cost stays real:** a live price-data dependency on a price-feature-free frozen model, a
   retrain + refreeze, and a wider serve-parity surface — for a benefit that is not reversal, is
   CPCV-marginal, and evaporates at the matched horizon.

**Recommendation: do NOT promote `st_rev` (nor low_vol/amihud/all3).** Per instructions I stopped
before wiring serve or retraining. This closes the reversal/low-vol/illiquidity Tier-0 line the
same way momentum closed: real-looking in one slice, not a stable model-level edge. The
price-feature harness stays for the next idea (residual reversal on sector-neutral returns, beta,
turnover), but raw short-term reversal should not be re-litigated in this universe.

## What was kept (default-off — shipped model / serve / paper-forward untouched)

- `panel.short_term_reversal()`, `panel.low_vol()`, `panel.amihud()` beside the momentum fns.
- `PRICE_FEATURES` widened to the 5-feature research registry; `price_feature_matrices(close,
  dv=None)` builds `amihud` only when dollar volume is supplied; `build_fundamental_panel`
  derives `rank_features` from the matrices actually built (a skipped amihud is never ranked).
- `scripts/run_reversal_test.py` (WF + CPCV head-to-head) and `tests/test_reversal_features.py`
  (PIT / parity / NaN-safety incl. zero-volume masking + dv-optional amihud). 214 tests green.

All of it is gated behind `price_features=False`: the frozen artifact, the serve path, and
the paper-forward book are byte-identical.

---

# News Memory & News-Aware Narration Verdict (2026-07-03)

Narration can now "bring up the past" — reference recent and historically-material news
themes and cite the article — WITHOUT weakening the anti-fabrication guarantee. A local,
timestamped news store (Intrinio headline+summary + versioned LLM extractions) feeds
number-free takeaways into the narration packet. **The whole layer is firewalled: news is
never a feature, never scored, never point-in-time-joined into the panel** — it is
live-view + narration ONLY. Everything is timestamped so it *could* be made point-in-time
later; it stays out of the signal (the honest fundamental OOS IC ~0.037 is the thing this
protects). Gate: the Phase-4 0-fabrication / full-traceability gate must still hold WITH
news context present. **GATE: PASS.**

## Faithfulness eval WITH news context (34 names × seeds 7+42 = 68 narrations, gemma4:26b)

`uv run python scripts/eval_narration.py --n 34 --news 5 --seed {7,42}` — 5 live Intrinio
headlines (number-free) attached to every name. Both seeds independently **GATE: PASS**.

| metric | seed 7 | seed 42 | combined |
|---|---|---|---|
| names carrying news context | 34/34 | 34/34 | **68/68** |
| fabricated numbers in FINAL output | 0/34 | 0/34 | **0/68** |
| citation traceability in FINAL output | 34/34 | 34/34 | **68/68** |
| narrations that cited ≥1 article | — | — | **66/68** (avg ~2.1; only 2 cited none) |
| first-pass valid (raw LLM) | 28/34 | 32/34 | **60/68 (88%)** |
| fixed by violation-feedback retry | 6/34 | 1/34 | 7/68 |
| template fallback (still grounded) | 0/34 | 1/34 | 1/68 |
| latency | mean ~95s | mean ~81s | — |

The guard earns its keep *with news present*: across the runs it caught fabricated numerals
(e.g. a spurious `-3.0`) and malformed / uncited citations — all fixed on retry, or degraded
to the deterministic grounded template (the single seed-42 fallback), so the reader never
saw a fabrication. Combined first-pass validity (88%) meets or beats the news-free Phase-4
run (82%): the added news contract costs no measurable regression, and news is genuinely
used (66/68 narrations cite at least one article).

## The firewall's linchpin: number-free takeaways

The packet's `context.news` carries `{id, date, source, event_type, takeaway}` — and the
takeaway is stripped of ALL numerals (`packet.news_context`). So the grounding guard's
numeral domain stays the *fundamentals* packet: a fabricated figure is still caught even
when a news summary contained that very number (a stripped-away "$873M" cannot re-enter via
narration — regression-tested). Only a date's YEAR survives, via the `date` field. The raw
article summary (the real numbers) lives in `news.sqlite` and the TUI, never in the packet.
News is attached AFTER model scoring in `serve.analyze(..., news=)`, and `context` is
excluded from the narration cache hash so live headlines never invalidate a cached read.

## What was built

- **Narrator contract** (`narrate/narrator.py`): the LLM MAY weave in news themes / past
  events but MUST cite the article — `{"id": "new_…", "direction": "reported"}`. News is a
  SEPARATE citation channel: it carries no supports/detracts sign (that would be the
  firewall leaking a signal into the score's framing), only the neutral "reported". Every
  cited news id must exist in the context; a numeral in the prose still traces to the
  packet exactly as before.
- **News store** (`src/stockscan/newsmem/store.py`, `news.sqlite`): raw articles
  (insert-or-ignore — ground truth, never mutated), extractions keyed (article_id,
  version), a per-company fetch throttle. `recall()` (keyword + structured: event-type /
  date / materiality) and `context_for()` (recent + notable-past) build the narration
  context.
- **Extraction** (`newsmem/extract.py`): headline+summary → `{event_type, entities,
  keywords, takeaway, sentiment, materiality}`, versioned + regenerable. Mirrors the
  narrator's discipline — an extraction may NOT assert a number not in the raw article
  (guard → one retry → deterministic heuristic fallback). `llm=None` yields the heuristic,
  so the store is always populated (the `--no-llm` path, tests).
- **Curation** (`newsmem/curate.py`): materiality floor + source credibility + title
  dedup. Press-wire sits below the credibility floor, so a wire item must be decisively
  material to surface — keep material, drop press-wire spam.
- **Ops** (`scripts/ops.py news [--no-llm] [--backfill PAGES]`): nightly watchlist-only
  ingest (idempotent, quota-capped by a 12h refetch throttle; light tier, heuristic under
  `--no-llm`), slotted into the nightly flow after `monitor`. Firewalled from the signal,
  so a degraded price night does not gate it. Lazy on ticker-open in the TUI, cached in the
  store; a lazy heuristic placeholder is upgraded in place by the nightly LLM run.
  `--backfill PAGES` paginates history (Intrinio `next_page`) to SEED the memory so
  recall's "notable past" has depth on day one instead of accruing over weeks.
- **TUI**: the ticker page's news section now shows the recalled memory (recent + notable,
  with event-type badges); the `n` key runs the local model with the recalled news attached
  (previously it silently produced only the template — now fixed).

## Deferred / accepted

- **Cached monitor narrations stay fundamental-only.** News is attached in the on-demand
  (`n` key) and serve paths, NOT in the nightly monitor's cached narration — so a cached
  read is stable and never misleadingly stale. Fresh-news narration is on-demand.
- **Paraphrased news themes aren't force-cited.** As with signals (Phase-4), the
  deterministic guard enforces that any news id the LLM *does* cite exists and that no
  number is fabricated; forcing a citation for an unnamed paraphrased theme needs an LLM
  judge (same deferred semantic check).
- Extraction quality is only as good as headline+summary (full-text is never fetched, by
  decision); the heuristic fallback is intentionally coarse.

Tests: **204 green** (+26 vs Phase-5's 178): 19 news-memory (extraction guard, curation,
store upsert/dedup/recall, ingest quota cache, heuristic→LLM upgrade, store→packet
firewall, historical backfill, watchlist-job orchestration) + 7 narration guards
(fabricated-number-with-news, news-citation existence / "reported"-only, number-free
context, cache-hash ignores news).

---

# Phase-5 — Continuous Operation & Paper-Forward Verdict (2026-07-02)

The machinery now runs unattended: idempotent ingestion jobs, a monitoring loop
with SQLite state, and an append-only paper-forward log against a model + thresholds
frozen TODAY. Gate (DESIGN.md §8): live behavior broadly consistent with backtest,
the machinery runs unattended. **GATE: PASS (operational; the live-vs-backtest
comparison is now accruing — first genuinely OOS month is July 2026).**

`uv run python scripts/ops.py {nightly|prices|fsds|universe|monitor|paper|health}`

## What was built (src/stockscan/ops/ + scripts/ops.py)

- **Idempotent ingestion jobs** (ops/jobs.py). Each is safe to re-run and logs a
  deltas dict to `ops_state.job_runs`:
  - **nightly prices** — a FULL-HISTORY refetch per ACTIVE column by security id,
    NOT an incremental append. Adjusted series rebase retroactively on every split
    or dividend (the vendor rescales all history), so grafting new bars onto stale
    history would manufacture a scale break at the seam — the exact artifact class
    Phase-3 fought. A security's history fits one API page, so a refetch costs the
    same as an increment and heals vendor revisions for free. Idempotent via the
    data: the reference column sets the session's target trading date, and any
    column already at that date is skipped without a request; a shrunken vendor
    response never replaces a fuller file (sanity guard); identical content is not
    rewritten (no mtime churn / cache rebuild).
  - **quarterly FSDS** — ingests any elapsed quarter missing from disk, then rebuilds
    the wide table. A not-yet-published quarter (404, or an opaque retry error on the
    newest quarter) is "waiting", not a failure; an OLD missing quarter is a real gap.
    Quarter parquets are now written atomically (tmp + os.replace) and a crash-damaged
    file counts as missing so it self-heals.
  - **universe refresh** — re-enumerates the Intrinio security master, diffs per-CIK,
    and applies changes in a crash-safe order: dead companies get a full multi-security
    re-splice under their NEW column name (OTC-afterlife securities only become
    candidates at death, so the death decline is captured, not truncated), renames
    rewrite the parquet's INTERNAL ticker column (the matrix pivots on it) keeping the
    fuller file on any target collision, new companies are fetched, and the universe
    parquet is replaced LAST as the commit point. Recently-dead columns keep getting
    refetched for a 120-day grace window (late OTC prints inside open forward windows).
- **Matrix cache** (panel.py) — the per-column store is 11k parquets that take ~11s
  to pivot; the nightly job persists the two wide matrices and the serve/monitor paths
  load them in <1s. Freshness is a manifest hash over sorted (filename, size, mtime),
  NOT bare mtime — os.replace renames preserve mtime, so an mtime check would call a
  renamed-column cache fresh. Cached == slow-path is asserted bit-identical.
- **Monitoring loop** (ops/monitor.py) — one pass over the watchlist: percentile-move
  alerts (|Δpct| ≥ 10 vs the last recorded state), new-filing detection from two
  sources (the wide table = numbers landed; EDGAR submissions = filed, numbers arrive
  next FSDS batch — both bootstrap-seed silently on first sight), and materiality-gated
  re-narration through the EXISTING cache (scan.py's pattern: analyze(llm=None) →
  narrate_smart). Percentile alerts are suppressed on a degraded price night (ranks
  over a half-updated store would fire false alerts, then fire them in reverse on
  recovery).
- **Paper-forward** (ops/paper.py) — the un-overfittable test, frozen TODAY:
  - `paper freeze` writes a write-once `baseline.json` (artifact content hash,
    trained_through, feature cols, frozen thresholds, the backtest expectations the
    live run is judged against, the degradation rule, and the KNOWN live-vs-backtest
    asymmetries) + the first entry in an append-only `vintages.jsonl`. Re-freezing the
    same artifact is a no-op; a different artifact hard-errors toward retrain-record.
  - `paper log` scores the live cross-section at the last COMPLETED month-end (the
    backtest's monthly grid) with the frozen artifact and appends an immutable
    `signals/<date>.jsonl` (a header with run metadata + cross-section stats + data
    vintage, then one line per name: score, pct, decile, top-decile, book membership,
    filing dates). Never overwrites; a re-run verifies the header hash and no-ops,
    re-applying the recorded hysteresis-book transitions (the book is reconstructed
    from the file, so SQLite and the append-only record cannot diverge across crashes).
    Refuses to run if the current artifact hash ≠ the latest registered vintage.
  - `paper compare` reads every logged month old enough to score, computes the realized
    forward return on RAW prices with 1/99 winsorization (the exact methodology behind
    the frozen baseline IC — see the code-review note below on why NOT scale-break
    repair), re-keys every logged name by CIK through the CURRENT universe map (a column
    renamed by death still resolves — the crashed names must not silently drop out), and
    reports rank IC / decile spread / book excess vs the frozen expectation. In-sample-
    flagged months are excluded from the gate metric.
  - `paper retrain-record` is the MANUAL, logged retrain event: it appends a new
    vintage. Nothing in the loop calls it; the quarterly-retrain cadence stays a human
    decision (DESIGN §10). The monitor and health both assert the running artifact is
    the registered vintage, so an in-place train_model.py overwrite is caught, not
    silently served.
- **No-retrain-in-the-loop is structural**: nothing under stockscan.ops imports fit /
  save_artifact / LGBMRegressor — enforced by a source-scan test.
- **Scheduling + health** — `ops.py nightly` is the single launchd entry (installed to
  ~/Library/LaunchAgents, daily 22:45): it runs prices → FSDS-if-due → universe-if-due
  → paper-log-if-due → monitor, each stage self-checking whether it is due, so a run
  missed while the Mac slept simply catches up next firing. A single repo-wide flock
  makes wake-coalesced double-fires and manual overlap harmless. `ops.py health`
  checks price/fundamentals freshness, matrix-cache sync, artifact-vs-vintage,
  baseline, paper cadence, job recency, and the LLM endpoint (informational — narration
  degrades to template by design); critical failures exit non-zero.

## First live run (2026-06-30 as-of, on real data)

The baseline is frozen at artifact vintage **b50bc6d9** (trained through 2026-03-31).
The first monthly log scored **2,970 liquid names** across 10 sectors, 297 top-decile,
595 in the initial long book. It is honestly flagged **in_sample: True** — the artifact
trained through March 2026 plus the 63-trading-day label horizon reaches the end of
June, so June is still inside the training information window. `compare()` excludes it
from the gate metric; the paper-forward comparison genuinely begins with the July log
(the first fully out-of-sample month), and it refuses to judge until 3 OOS months have
accrued. The live label is measured on RAW, unrepaired prices with 1/99 per-month
winsorization — EXACTLY how the frozen baseline IC was computed (build_fundamental_panel
uses load_matrices directly; winsorization, not scale-break repair, tames the artifacts
on the label side). The delisted BBBY watchlist name flows through the monitor and is
correctly reported as a lapsed filer (last 10-K > 550d stale), no crash.

## The adjusted-price-rebase decision (why full-refetch, not append)

An adversarial review of the plan BEFORE coding (3 lenses, then this same review on the
shipped code) flagged as its top finding that a 7-day-overlap incremental merge is
structurally wrong for back-adjusted prices: a single split or dividend rebases the
whole vendor history, so the retained pre-overlap rows sit on the old scale and the
seam prints a fabricated return (a 1:10 reverse split → fake +900%; every dividend →
a small persistent bias) — and nothing on the serve/monitor/paper path repairs it. The
shipped nightly job refetches full history per active column instead; a regression test
simulates a rebased vendor response and asserts the healed series has no seam.

## What the code review caught (fixed + regression-tested)

A 3-lens adversarial review of the SHIPPED code (find → adversarially verify each
finding; 10 agents) confirmed 7 defects, all fixed with regression tests:

- **[critical] `compare()` was flattering the live track** — it applied the
  backtest's NAV-side price hygiene to the LABEL, which masks sub-penny death
  prints to NaN and lets `forward_return_to_last`'s ffill carry the pre-crash price
  forward, fabricating a ~0% return for a name that actually died to zero (empirically:
  a name crashing to sub-penny read −99.99% raw, +0.0% after hygiene). It also diverged
  from the baseline, whose IC was measured on raw+winsorized prices. Fixed: the label
  is now raw + 1/99 winsorization, matching the baseline exactly; a regression test
  asserts a mid-window death keeps its ~−99% loss.
- **[critical] `build_fundamentals_wide` wrote the read-hot serve file
  non-atomically** — a crash mid-COPY would pin a truncated `fundamentals_wide.parquet`
  that every serve/monitor/paper pass reads, with no self-heal (the quarter files still
  look ingested). Fixed: tmp + os.replace, mirroring the per-quarter builder.
- **[major] A transient failure of the price-refresh reference column stranded the
  whole universe** — if the AAPL heartbeat fetch failed, its stale on-disk date became
  the session target, short-circuiting all ~11k columns to "fresh" while the job
  reported success. Fixed: the target is adopted only when the reference bar is
  actually current; otherwise the rest is fetched normally and the run flags itself
  degraded (`reference_ok`).
- **[major] The nightly paper-log couldn't backfill a multi-month outage** — a Mac
  asleep for weeks would lose the middle months permanently. Fixed:
  `missing_paper_months` enumerates every completed month-end since the freeze with no
  file and logs each oldest-first (log_signals is PIT at as_of, so a late run scores
  identically); the manual `paper log` and the health check both use it.
- **[minor] The matrix cache stopped rebuilding after a universe refresh deleted its
  manifest** — the nightly rebuild only fired on a write, so the serve path was pinned
  to the slow load. Fixed: rebuild whenever the cache is stale/missing, not only on a
  write.
- **[minor] A degraded price night suppressed alerts but not narration** — a jittered
  cross-section could still cache a full narration against a wrong percentile and reset
  the materiality baseline. Fixed: a degraded night forces template-only narration
  (which never caches).
- **[minor] A paper-log no-op was recorded as job status "ok"** rather than "noop".
  Fixed: `_run_logged` honors a returned status.

The review's top finding matched the pre-coding plan review's: an incremental
adjusted-price merge is unsound (the full-refetch design was the right call).

## Data-layer backlog (partially closed)

The store CAPTURES unadjusted close/volume (uclose/uvolume) alongside the adjusted
OHLCV — every fetch carries them, readers tolerate the mixed schema (union_by_name),
and `scripts/backfill_unadjusted.py` refetches ONLY the pre-schema files, resumably.
The full backfill is now **COMPLETE (2026-07-02)**: all 11,029 price columns carry
populated uclose/uvolume (audited: 0 missing, 0 all-null; 1 single-row name lacks
uvolume), and the wide matrix cache was rebuilt on top of it. Still deferred,
deliberately: the actual switch of the liquidity floor from adjusted to unadjusted close
is a re-baseline event (it changes historical universe membership → panel rebuild →
retrain → new artifact vintage → re-freeze). The raw data is now available store-wide;
the migration remains a logged vintage step — a human decision — not a silent threshold
change under the frozen baseline.

## Deferred / accepted

- The FSDS quarterly publication lag makes every live month-end miss the freshest 0-3
  months of filings that its backtest twin had; frozen into baseline.json as a known
  asymmetry and stamped (data_vintage) in every run header, not re-discovered as decay.
- compare() is gross, close-to-close, cost-free — gated on the like-for-like IC/spread
  pair; the NAV net-excess numbers are context.
- Value features (PIT market cap), 10-Q cadence, and the distress head remain deferred
  from earlier phases.
- The nightly job refetches the full active universe against Intrinio every night; on a
  metered plan this is real quota. Uninstall with `ops.py install-launchd --uninstall`.

Tests: 160 green (110 Phase-4 + 50 new: ops state/jobs/paper/monitor/health, the
matrix cache, and the template-cache and price-schema regressions).

---

# Phase-4 — Narration Hardening Verdict (2026-07-02)

The NARRATE stage is now a constrained, validated, cached pipeline on real local
models. Gate (DESIGN.md §8): ~0 fabricated numbers, full citation traceability, a
sector scan in an acceptable window. **GATE: PASS.**

## Faithfulness eval (34 real tickers, gemma4:26b, seeds 7 + 42)

| metric | result |
|---|---|
| fabricated numbers in FINAL output | **0 / 34** |
| citation traceability in final output | **34 / 34** |
| first-pass valid (raw LLM) | 28 / 34 (82%) |
| fixed by violation-feedback retry | 6 / 34 |
| template fallback needed | **0 / 34** |
| latency | mean ~153s, p90 ~270s (incl. retries) |
| cold top-10 narration backfill | ~25 min (lazy/async per §7; ~12 min on the phi4 light tier; ~0 cached) |

The guard earns its keep visibly: DRH's first attempt fabricated the numeral 31 —
precisely the date-component class this phase's grounding fix closed — the
validator caught it, the retry fixed it, the reader never saw it. One 30-name run
also crash-tested the harness itself (an over-300s generation propagated a raw
transport timeout at name 29): LLM errors now degrade to violation → retry →
template instead of crashing, with a regression test. 110 tests green.

## What was built

- **SHAP drivers** (`Artifact.explain`, LightGBM native `pred_contrib` — an EXACT
  decomposition, row-sums equal the score): top-5 signed contributions enter the
  packet as `model.drivers`, namespaced `driver:<id>` because the model's learned
  direction legitimately disagrees with the textbook direction (that IS the
  learned-signs edge — e.g. high leverage as a positive model driver while the
  signal reads it as a weakness).
- **Cited-JSON contract** (narrator.py): the LLM returns `{reasoning, summary,
  citations:[{id, direction}]}`. The deterministic validator enforces: every
  numeral in summary AND reasoning grounded; every citation id exists; every
  direction agrees with the packet's own `read` (signals) or SHAP sign (drivers);
  any signal/driver MENTIONED by name must be cited (the guard is not opt-out);
  the 45-55 effective-percentile band accepts either direction. Violations →
  one retry WITH the violation list → deterministic template fallback.
- **The packet carries the verdict**: each signal now has an explicit
  `read: supports|detracts` computed in code — the LLM copies it, never derives
  it. This single change (plus a brace-balanced JSON parser) took gemma4:26b from
  0/2 first-pass valid to 3/3 in the confirmation round. Deterministic numbers,
  LLM prose — the founding principle, applied to directions too.
- **Materiality-gated cache** (SQLite): unchanged packet → cached; minor drift →
  light 14B tier; new filing / ≥10-pctile move / changed top drivers → full tier.
  Volatile fields (as-of stamps, exact score) are excluded from the change hash so
  daily re-queries can actually hit; the materiality baseline only resets on FULL
  narrations (no ratchet past the threshold via small steps).
- **Sector scan** (`scripts/scan.py`): the deterministic ranked table renders in
  ~37s (data load dominated); narration is lazy for the top N, cache-aware.

## The model decision (benchmarked on the M5 Pro, DESIGN §10's deferred call)

| model | warm latency | tok/s | first-pass valid (after fixes) |
|---|---|---|---|
| **gemma4:26b (full tier)** | ~150s | 32.8 | 3/3 |
| **phi4 (light tier)** | ~74s | 11.9 | 3/3 |
| mistral-small3.1 | 72s | 7.5 | not selected |
| gpt-oss:20b | 106s | 28.6 | reasoning-token heavy |
| qwen3.6:27b-mlx | **timed out repeatedly** | — | disqualified |

Runtime verdict: **llama.cpp/GGUF via Ollama**. The MLX-format path was
non-functional on this Ollama build (hard timeouts even warm) — DESIGN's
MLX-vs-llama.cpp question resolves itself empirically. The DESIGN-era candidates
(Qwen2.5-32B / Gemma-3-27B) were superseded by newer local equivalents already on
disk; no new downloads were needed.

## What the adversarial review caught (fixed + regression-tested)

The review's finder agents produced empirical repros; its verify phase died on an
API spend limit, so every finding was re-verified inline before fixing:

- **Date components blessed fabrications**: the Phase-2 date fix whitelisted bare
  month/day integers, so "up 12%" or "31% share" passed grounding for any Dec-31
  filer. Dates (ISO and natural-language) are now stripped from BOTH sides with
  only the year surviving as a numeral.
- **The direction guard was opt-in**: a wrong-direction claim could simply omit
  its citation. Now any signal/driver mentioned by its packet label without a
  covering citation is a violation.
- **Brace-naive JSON parsing** falsely rejected valid chatty replies (a `{` in
  surrounding prose corrupted the slice) — this, not model quality, drove much of
  the initial 0/2 benchmark validity. Now brace-balanced and string-aware.
- Template fixes: honest grounded flag (checked, not asserted), no scientific
  notation leaking untraceable mantissa/exponent numerals, no signal listed as
  both strongest and weakest on thin packets.
- Cache fixes: materiality checked before the unchanged shortcut; light-tier runs
  no longer ratchet the baseline.

## Deferred / accepted

- Direction enforcement matches mentions by exact packet labels — paraphrased
  references ("profitability" for ROA) escape the mention check (the citations
  the LLM does provide are still direction-checked). A semantic-level check needs
  an LLM judge — Phase-5 faithfulness monitoring candidate.
- Latency: ~150s/name full tier is fine for watchlist monitoring (narrations are
  cached and materiality-gated); a cold full-sector scan of hundreds of names
  remains intentionally lazy/async per DESIGN §7.
- `scripts/analyze.py` re-narrates per invocation (no cache wiring in the CLI yet);
  scan.py is the cached path.

---

# Phase-3 — Backtest & Signal Mechanics Verdict (2026-07-02)

**GATE: PASS.** The net-of-cost edge survives real trading mechanics. Signals are the
purged walk-forward OOS model scores (never the in-sample artifact); execution is
next-bar open with liquidity-scaled costs; the whole thing runs on the honest
no-impute panel. `uv run python scripts/run_phase3.py [--cpcv]`.

## Headline (2013-08 → 2026-03, 152 monthly rebalances, ~2,600 liquid names)

| book | net CAGR | gross | Sharpe | maxDD | turnover |
|---|---|---|---|---|---|
| universe EW (benchmark) | +7.79% | +8.80% | 0.36 | −42% | 1.1x/yr |
| **long-only, hysteresis** | **+9.26%** | +10.52% | **0.42** | −42% | 1.6x/yr |
| long-only, hard decile | +6.93% | +9.16% | 0.29 | −44% | 3.0x/yr |
| long/short, hysteresis | −1.29% | +1.52% | −0.13 | −36% | 3.9x/yr |

- **Long-only beats the universe by +1.48%/yr net**, and stays ahead at 2x costs
  (+8.02%). A modest, real, capacity-limited edge — consistent with the IC story.
- **Hysteresis is load-bearing**: enter-20%/exit-40% beats the hard decile by
  +2.33%/yr net while nearly halving turnover. Signal decay is slower than monthly
  churn; trading less is worth more than holding the sharpest tail.
- **The short book dies of borrow + costs, exactly as §6 predicted**: +1.52% gross
  → −1.29% net (−1.75% at 2x borrow). Verdict per the §6 rule: **drop the short
  book**; the product signal is long-tilt only.
- **Where the edge lives**: decile spread +4.3%/qtr in $1-5M-ADV names, +1.8% in
  $5-25M, +0.5% in >$25M — positive everywhere (gate passes) but strongly
  small-cap-tilted. For a personal-scale account this is fine (its capacity IS
  personal-scale); it is not an institutional strategy claim.
- IC by sector bucket: financials +0.030 (t 2.3), non-financials +0.043 (t 6.1).
- **CPCV distribution** (45 purged combinations): mean IC +0.037, 5th pct +0.013,
  100% of combinations positive. **PBO (CSCV, 12,870 splits)**: 0.03 over the
  six long-only variants (the family we actually select from — gated), 0.26 over
  all 12 trials including the structurally-losing L/S half.

## What real data broke (and how it was fixed)

The first full run produced garbage that LOOKED like alpha: universe EW at +32%
CAGR / 122% vol, one +424% portfolio day. Diagnosis in two acts:

1. **Sub-penny quantization**: adjusted closes are stored to 4 decimals, so junk
   names print 0 → 0.0001 (= +inf%). 207,661 prints masked below a $0.01 trust
   floor — crashes into the floor are taken, bounces inside it never compound.
2. **Vendor scale breaks**: some dead names' adjusted series jump scale mid-stream
   (Shineco printed 11.09 → 137,160.00 overnight on ~$100k volume — a mis-applied
   corporate action). My first fix froze series at the break — and the adversarial
   review workflow proved that WRONG in the dangerous direction: the same threshold
   caught Tricida's real −94.5% trial-failure day, so a held position would have
   exited at the pre-crash price (look-ahead in our favor). The shipped fix
   REPAIRS instead of erasing: a one-day ratio beyond anything real (>20x up,
   <−96% down, consecutive prints only so delisting-splice gaps stay real) has its
   log-return zeroed and the series rebuilt at a consistent scale. 1,677 break
   days across 192 series repaired; Tricida-class real crashes hit the NAV in full.

Other review catches (36-agent adversarial workflow; all fixed + regression-tested):
a missing open print caused a costless phantom exit at the stale prior close (now
fills at the same-day close, with cost); traded notional compared unnormalized
drifted weights (costs were understated whenever the window return was nonzero); a
short squeeze could take NAV negative and NaN the summary (now liquidates to zero);
signals predating the price history wrapped to end-of-sample liquidity; the NAV
series lacked its inception point; PBO silently passed NaN-lambda combinations and
was structurally flattered by the losing L/S trials (now reported per family and
gated on long-only); test fixtures had open==close everywhere so open-fills were
untested.

## Honest caveats

- The label-side diagnostics (IC, bucket spreads) use unrepaired prices (their 1/99
  winsorization already clips artifacts); the NAV uses repaired prices. A unified
  data-layer repair (and ideally re-fetching unadjusted prices) is the clean fix —
  backlog for the data layer, measured here rather than hidden.
- Liquidity floors still run on adjusted closes (Phase-2 note stands).
- Costs/borrow are modeled schedules (tiered by ADV), not broker quotes; the 2x
  stress bounds the assumption.
- Scores are walk-forward OOS, but strategy-variant selection (hysteresis bands)
  is in-sample across the whole period — that's what the PBO=0.03 addresses.
- No distress-flag hard exits yet (needs the distress head, deferred).

## Deferred

- Phase 4: narration hardening (LLM serving, SHAP top-k, faithfulness eval).
- Phase 5: monitoring loop, paper-forward vs backtest, artifact vintages.
- Data layer: unadjusted price fetch (liquidity floors + repair at source),
  value features via PIT market cap, 10-Q cadence, distress head.

---

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
