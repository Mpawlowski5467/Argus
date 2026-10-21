# EDGAR full-text search backend (efts.sec.gov) — as observed 2026-10-20

Undocumented JSON backend behind the public UI at `https://www.sec.gov/edgar/search/`.
Sources, in order of authority: the backend's own `query` echo (R4), the four recorded
responses (R1–R4, `edgar_fts.har`), and the UI's `js/edgar_full_text_search.js`
(fetched once; line numbers below refer to that file as fetched).

## How it was captured

- REA's headless-browser scenario against the UI was refused **403** twice by www.sec.gov
  ("Request Rate Threshold Exceeded — automated access … must comply with SEC.gov's Privacy
  and Security Policy"): SEC blocks undeclared automated clients, and the headless browser
  cannot send a declared contact User-Agent. Not retried.
- Everything else went through the project's throttled `EdgarClient` (≤8 req/s, declared UA):
  the UI page + its JS (2), four backend searches (R1–R4), one `data.sec.gov` submissions
  read (R5) — 7 requests, all 200. Recorded as `edgar_fts.har` (UA contact scrubbed) and
  analysed offline with REA (`edgar_fts.har.evidence.json`, `edgar_fts.api_surface.json`):
  one endpoint, `GET https://efts.sec.gov/LATEST/search-index`, 4/4 records 200 JSON.

## Request

`GET https://efts.sec.gov/LATEST/search-index?<params>` — plain query string, no auth, no
cookies, JSON response. The UI sends it with `$.ajax({data: searchParams, type: 'GET'})`
(JS L1081).

| param | form | meaning (backend translation from the R4 `query` echo where observed) |
|---|---|---|
| `q` | text | Full-text query on `doc_text`. `"quoted phrase"` → `match_phrase`; `A OR B` → `bool.should` with `minimum_should_match: 1` (**verified R4**) |
| `forms` | `8-K` or `10-K,10-Q` | `terms` on **`root_forms`** — so `8-K` also returns **8-K/A** (R1: 41 of 200 hits were 8-K/A) (**verified**) |
| `ciks` | `0001948864,0001407583` | 10-digit zero-padded, comma-separated → `terms` on `ciks` (**verified R3, R4**) |
| `startdt`, `enddt` | `YYYY-MM-DD` | inclusive `range` on `file_date` (**verified R4**). The UI also sends `dateRange=custom` (presets `30d`/`1y`/`5y`, default `5y`) |
| `category` | `custom` / `all` / `form-catN` | UI form-group selector; `custom` + `forms` is the explicit form list (JS L766) |
| `from` | int | result offset; the UI derives it as `(page − 1) × 100` (JS L1062, `resultSize = 100` L1641). Page size is fixed at **100** (**verified R2**) |
| `items` | `4.02,3.01` | 8-K item filter, sent by the UI from its *items* box (JS L744/759). **Not exercised against the backend here** |
| `entityName` | display string | UI convenience; the CIK is extracted from it client-side (`extractCIK`, JS L938) |
| `sics`, `locationCode(s)`, `locationType` | codes | industry / state filters (unused by the design) |
| `filter_forms`, `filter_entityName`, `filter_location` | | facet "breadcrumb" filters (unused) |

Not needed and not sent by the design: `fileType`, `page` (UI-only; the backend takes `from`).

## Response (Elasticsearch-shaped)

```
{ took, timed_out, _shards,
  query:        <the ES query actually run — useful as a contract check>,
  hits: { total: {value, relation}, hits: [ {_id, _score, _source}, … ] },
  aggregations: { entity_filter, form_filter, sic_filter, biz_states_filter } }
```

- **One hit = one document, not one filing.** `_id` is `"<accession>:<filename>"`; an 8-K
  whose press-release exhibits also match appears several times (R1+R2: 200 hits → 172
  accessions, max 4 per accession). Dedupe on `_source.adsh`.
- `_source` keys (all four responses): `adsh, ciks[], display_names[]` (e.g.
  `"NAME  (TICKER)  (CIK 0001948864)"`), `file_date` (`YYYY-MM-DD`), `period_ending`,
  `form` (`8-K`/`8-K/A`), `root_forms[]`, `file_type` (`8-K`, `EX-99.1`, …),
  `file_description`, **`items[]`** (8-K item codes, e.g. `["1.01","2.01"]`), `sequence`,
  `sics[]`, `biz_locations[]`, `biz_states[]`, `inc_states[]`, `file_num`, `film_num`, `xsl`.
  `doc_text` is excluded from `_source`, so the response says **which document** matched
  but **not which phrase** — per-phrase attribution needs one query per phrase.
- Ordering is by **relevance (`_score`), not date** — a nightly reader must page to the end
  (`from += 100` until `from >= total.value`), not stop at "older than last run".
- `total.relation` was `eq` at 531. Elasticsearch's default 10,000-result window
  (`from + size ≤ 10000`) very likely applies — **not verified**; irrelevant at nightly
  volumes (R3: 5 hits/7 weeks for one name; R4: 14 for two names, five phrases).
- Latency: 139 ms – 1.4 s per request (`took` up to ~1 s server-side on the broad query).

## data.sec.gov submissions — the 8-K item codes for free

`/submissions/CIK##########.json` (already fetched nightly per watched CIK by
`ops/monitor.py:detect_edgar_filings`) carries `filings.recent.items` per filing as a
comma-joined string (R5: `"1.01,5.02,9.01"`), parallel to `accessionNumber`, `form`,
`filingDate`. Structured 8-K events — **Item 3.01** (notice of delisting / failure to
satisfy a listing rule), **Item 4.02** (non-reliance on previously issued financials),
**Item 4.01** (auditor change) — can be detected with **zero** additional requests.

## Caveats

- Undocumented backend: no versioning promise. The design treats it as fail-open and
  contract-tests the request it builds against the recorded responses.
- Index lag between EDGAR acceptance and full-text availability is not documented; the
  design overlaps its nightly window by a few days and relies on accession dedupe.
- SEC fair access: declared UA + ≤10 req/s per IP (the client stays at 8). Never from a
  headless browser.
