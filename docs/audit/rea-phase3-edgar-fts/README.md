# REA Phase 3 — EDGAR full-text search backend + 8-K alert design (2026-10-20)

| file | what |
|---|---|
| `efts_api.md` | the backend contract: endpoint, every query parameter, response shape, caveats |
| `DESIGN-8k-filing-flags.md` | the nightly 8-K flag design for `ops/monitor.py` — approved + implemented (notes at the end) |
| `edgar_fts.har` | the 7 recorded requests (UI page + JS bodies omitted; 4 backend searches + 1 submissions read with bodies), declared-UA contact scrubbed |
| `edgar_fts.har.evidence.json` | REA `inspect-web-network-capture` Evidence over that HAR |
| `edgar_fts.api_surface.json` | REA `api-surface` view: one backend endpoint, 4/4 records 200 JSON |

SEC traffic for this phase: 9 requests in total — 2 from REA's headless browser (both
refused 403: SEC blocks undeclared automated clients) and 7 through
`stockscan.edgar.client.EdgarClient` (≤8 req/s, declared UA), all 200. No logged-in or paid
sites. The backend searches used phrases + CIKs taken from public results, never the
owner's watchlist or holdings.
