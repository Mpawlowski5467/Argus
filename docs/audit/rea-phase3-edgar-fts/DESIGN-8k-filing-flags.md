# Design proposal — nightly 8-K filing flags for watched + held names

Status: **APPROVED 2026-10-20 with the proposed defaults; IMPLEMENTED** on `rea-audit`
(uncommitted) — see "Implementation notes" at the end. Backend facts it relies on are in
`efts_api.md` (recorded 2026-10-20).

## Goal

Each night, flag new 8-Ks (incl. 8-K/A) from **watchlist ∪ held** names that signal
accounting or listing trouble, and surface them in the Watch view's alert list. Display /
alert only — firewalled from the verdict exactly like the risk heads (Phase 2).

## Two detectors, one alert stream

| detector | source | cost per night | catches |
|---|---|---|---|
| **A. 8-K item codes** | `filings.recent.items` in the submissions JSON the monitor **already** fetches per CIK | **0 new requests** (+1 per held-but-unwatched name, see Scope) | Item **4.02** non-reliance · Item **3.01** delisting notice / listing deficiency · (optional) Item **4.01** auditor change |
| **B. full-text phrases** | `efts.sec.gov/LATEST/search-index`, `forms=8-K`, `ciks=<all names>`, date window | **1 request per rule** (+1 per extra page of 100; CIK list chunked at 100) — ~5/night | the requested phrases anywhere in the 8-K **or its exhibits** |

Default rules (config, owner to confirm — see decisions):

| rule id | detector | match | tier |
|---|---|---|---|
| `nonreliance_402` | A | item `4.02` | critical |
| `delisting_301` | A | item `3.01` | critical |
| `going_concern` | B | `"going concern"` | critical |
| `non_reliance` | B | `"non-reliance on previously issued"` (the Item 4.02 caption; narrowed 2026-10-20, see notes) | critical |
| `material_weakness` | B | `"material weakness"` | flag |
| `restatement` | B | `"restatement"` | flag |
| `delisting_notice` | B | `"delisting notice" OR "notice of delisting" OR "deficiency letter"` | flag |

One EFTS query per rule (not one OR query) because the backend returns *which document*
matched but not *which phrase* (`doc_text` excluded) — per-rule queries give attribution
for ~4 extra requests a night. A rule's own synonyms are OR'd inside its `q` (verified
semantics: `should` + `minimum_should_match: 1`).

## Where it lives (as requested: `ops/monitor.py`)

- `edgar/client.py`: `EFTS_HOST = "https://efts.sec.gov"` and
  `EdgarClient.full_text_search(q, *, forms, ciks, startdt, enddt, from_=0) -> dict` — builds
  exactly the documented params (`forms`, 10-digit comma `ciks`, ISO dates, `from`) and goes
  through the existing `_get` (limiter, declared UA, retries). `edgar` is core; this adds no
  forbidden import, so `audit_firewall.py` stays clean.
- `ops/monitor.py`:
  - `alert_universe(state) -> list[int]` = watchlist ∪ positions with `shares > 0`.
    `run_monitor` computes it **before** its `watchlist empty` early return (today a
    held-only book gets no filing checks at all — monitor.py:160).
  - `detect_edgar_filings` refactor: fetch each submissions JSON **once**, hand `recent` to
    both the existing 10-K/10-Q path (unchanged, still watchlist-only so `filing_detected`
    noise does not grow) and the new `detect_item_flags(recent, rules)`.
  - `search_text_flags(client, ciks, rules, startdt, enddt)` — per rule: page with
    `from += 100` until `from >= total.value` (results are relevance-ordered, so no early
    stop on dates); collapse exhibit-level hits to one row per `(cik, adsh)` keeping the
    matched documents (`file_type`) and `items`.
  - `run_filing_flags(state, client, rules, today)` — orchestrates A + B, dedupes, writes
    alerts, returns a deltas block; called from `run_monitor` (new `filing_flags=True` arg,
    mirrored as `--no-filing-flags` in `ops.py monitor`).

## Window, dedupe, bootstrap

- **Window**: `startdt = cursor − 3 days` (index-lag overlap), `enddt = today`; cursor kept
  in `kv["filing_flags_cursor"]` and advanced **only when every rule's search succeeded**.
  First run / missing cursor: last 7 days.
- **Dedupe table** (new, `create table if not exists`):
  `filing_flags(cik, accession, rule, source, form, file_date, docs, first_seen,
  primary key (cik, accession, rule))`. `known_filings` is not reused — its key has no
  accession, so two 8-Ks on one day would collide.
- **One alert per (cik, accession)** at first sight, listing every rule it matched that
  night; rules found later for an already-alerted accession are recorded, not re-alerted.
- **Bootstrap** (name newly entering the universe): matches are seeded silently **except**
  those filed within the last 7 days, which do alert (decision #1 below).

## The alert

- kinds: `eightk_critical` (any critical rule matched) and `eightk_flag` (flag-tier only).
  `eightk_critical` joins `notify.HIGH_SEVERITY` (macOS banner); `eightk_flag` stays in-app.
  Neither goes in `ROUTINE_ALERT_KINDS` — both count in the badge and the morning brief.
- message (plain text — the Watch table escapes it):
  `AAPL (cik 320193): 8-K/A filed 2026-10-14 — items 4.02, 9.01; text: "going concern" in
  EX-99.1 [filing flag — not a trade signal]` + the existing `_held_note` (≈x% of your book).
- payload: `{accession, form, file_date, rules[], items[], docs[], url}` with
  `url = https://www.sec.gov/Archives/edgar/data/{cik}/{accession-no-dashes}/{accession}-index.htm`.
- Watch view: works unchanged (alerts render `kind` + `message` generically, app.js:1079).
  Optional follow-up: render `payload.url` as a link.

## Firewall (same contract as Phase 2)

- Writes **only** `filing_flags` rows and `alerts`. Never `signal_state`, never the packet,
  `analyze()`, narration, chat/panel context, the paper book, or any score.
- Locked by tests: (1) `run_monitor` with and without flag hits → identical
  `signal_state` rows; (2) Phase 2's proposed `tests/test_web_firewall.py` book-perturbation
  test gains an `eightk_critical` alert → `/api/ticker` stays byte-identical;
  (3) `audit_firewall.py` stays CLEAN.

## Failure handling

Fail open, per rule: an EFTS error (retries exhausted, non-JSON, schema drift) records
`{"rule", "error"}` in the monitor deltas, emits no alerts for that rule, and **holds the
cursor** so the next night re-covers the window. A health check `filing_flags` warns when
the stage has errored three nights running or the cursor is > 4 days stale — the backend is
undocumented, so silence must be visible. Kill switch: `STOCKSCAN_8K_FLAGS=off`.

## Tests — recorded fixtures only, never live SEC

- `tests/fixtures/edgar_fts/`: trimmed R1/R3/R4 bodies (shape preserved, 2–4 hits) + R5
  submissions `recent` arrays, plus synthetic variants: `total=230` across three pages,
  one accession matched in three exhibits, an 8-K/A, an empty result, a 500.
- A `FakeEdgarClient` serves fixtures by URL and **records every URL**; `httpx` is
  monkeypatched to raise, so any live call fails the test.
- Cases: param construction (phrase quoting, `forms=8-K`, 10-digit comma `ciks`, ISO dates,
  `from` paging); paging stops at `total`; exhibit collapse → one alert; 8-K/A included;
  item codes parsed from `"1.01,4.02,9.01"`; idempotent across runs; bootstrap 7-day rule;
  cursor held on failure; fail-open deltas; held-but-unwatched names included; the three
  firewall locks above. `uv run --extra dev --extra web pytest -q` stays green.

## Owner decisions before implementation

1. **Bootstrap**: alert on ≤7-day-old matches for a newly watched/held name (proposed), or
   seed everything silently like the 10-K path?
2. **Tiers / banners**: is the critical set (4.02, 3.01, "going concern", "non-reliance")
   right for macOS notifications?
3. **Phrases**: keep the literal five, or the widened `delisting_notice` synonyms above?
   Add item **4.01** (auditor change) as a flag-tier rule?
4. **Privacy**: each night the EFTS query sends the full watch+held CIK list to SEC in one
   URL (today the submissions fetch already reveals the same names, one per request).
   Acceptable?
5. **Calibration first?** Optionally ship with a `ops.py filing-flags --dry-run --since 90d`
   preview so you can see what the rules would have flagged on your book before alerts go live.

## Implementation notes (2026-10-20)

Owner decisions taken: (1) new names alert on matches ≤7 days old, older ones are stored
silently; (2) the critical tier raises a macOS banner (`eightk_critical` in
`notify.HIGH_SEVERITY`); (3) the widened delisting phrases and item 4.01 (flag tier) are in;
(4) the privacy trade-off is accepted; (5) the `--dry-run` preview ships.

| piece | where |
|---|---|
| rules, window, kill switch | `config.py` — `FILING_FLAG_RULES`, `FILING_FLAG_*`, `STOCKSCAN_8K_FLAGS`, `HEALTH_FILING_FLAGS_DAYS` |
| backend query | `edgar/client.py` — `EdgarClient.full_text_search` |
| store | `ops/state.py` — `filing_flags` (PK cik, accession, rule) + `filing_flag_ciks` |
| detectors + pass | `ops/monitor.py` — `alert_universe`, `fetch_submissions` (one fetch shared with the 10-K path), `detect_item_flags`, `search_text_flags`, `filing_flag_window`, `run_filing_flags`; `run_monitor(filing_flags=, client=, today=)` |
| banner / health | `ops/notify.py` (`eightk_critical`), `ops/health.py` (`filing_flags_freshness`) |
| CLI | `ops.py monitor --no-filing-flags`, `ops.py filing-flags`, `ops.py filing-flags --dry-run --since 90d` |
| tests | `tests/test_filing_flags.py` (18, recorded fixtures in `tests/fixtures/edgar_fts/`, httpx disarmed) + `tests/test_web_firewall.py` (4, HTTP firewall incl. an 8-K alert) |

Window detail settled during the build: the nightly window is always at least the last 7
days (not just "cursor − 3"), so the bootstrap rule for new names falls out of the window
itself; item-code matches outside the window are stored silently, which also keeps a rule
added later from alerting on years of history.

Verification: 510 tests pass (488 → +22); two leak/bootstrap mutants are caught (the
first leak mutant survived a signal-only assertion — the firewall test now also requires
every non-8-K alert to be identical with the pass on and off). Live `--dry-run --since 90d`
on the throwaway demo book (5 names): 10 SEC requests, ~2 s, nothing written.

**Finding from that preview — resolved:** the one match was NFLX's 2026-07-22 8-K, raised
*critical* for `"non-reliance"` found in **EX-5.1**, a legal-opinion exhibit (boilerplate,
not a restatement). One universe-wide request for the Item 4.02 caption phrase
`"non-reliance on previously issued"` over the same 90 days: 35 hits, 27 of them genuine
item-4.02 8-Ks. **Owner decision (2026-10-20): narrow `non_reliance` to that caption, keep it
critical.** Pinned by `test_non_reliance_rule_searches_the_item_402_caption`.
