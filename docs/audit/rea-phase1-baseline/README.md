# REA Phase 1 — end-to-end web capture baseline (2026-10-18)

A recorded run of the core argus-web flow against a **throwaway demo instance**, kept as
the baseline later changes are diffed against. Captured with REA 6.5.0
(`capture-browser-scenario`, Playwright, Chrome for Testing 153, headless, 1280×720) on
branch `rea-audit` at `e137f8b`; served as-of 2026-09-14, vintage `b50bc6d9f4bea56a`.

**Flow** (`scenario.json`, 23 declared steps): scan loads → type `AAPL` in search → open
AAPL → *Generate local-model read* (narration) → *run analyst panel* (bull/bear/risk/
synthesis) → Ask "why is it ranked here?" → Book → Markets → Paper → Watch.

## Files

| file | what |
|---|---|
| `scenario.json` | the REA scenario input — re-run it unchanged so step IDs align |
| `scenario.evidence.json` | REA Evidence record (CLI `--full-output` envelope): per-step outcomes, every network/console event, final URL, storage, screenshot |
| `timing.jsonl` | one line per HTTP request from the loopback timing proxy: duration, TTFB, request/response bytes (REA scenario events carry no timings) |
| `requests.md` | the two joined: step table, every request, slowest calls, repeated routes, console errors |
| `final-watch.png` | end-of-run screenshot (demo book only) |
| `demo_setup.sh` | rebuilds the throwaway instance (isolation details inside) |
| `timing_proxy.py` | the stdlib timing proxy (`:8025 → :8024`) |
| `analyze_run.py` | joins Evidence + timing log into `requests.md` |

## Re-capture and diff

```bash
bash docs/audit/rea-phase1-baseline/demo_setup.sh /tmp/argus-demo
source /tmp/argus-demo/demo.env.sh && uv run --extra web python scripts/argus_web.py      # :8024
python3 -I docs/audit/rea-phase1-baseline/timing_proxy.py 8025 8024 /tmp/argus-demo/timing.jsonl
rea capture-browser-scenario docs/audit/rea-phase1-baseline/scenario.json --format json --full-output > /tmp/argus-demo/after.json
python3 -I docs/audit/rea-phase1-baseline/analyze_run.py /tmp/argus-demo/after.json /tmp/argus-demo/timing.jsonl > /tmp/argus-demo/requests.md
```

Structural diff (which requests each step made, statuses, console, final screenshot/
storage): extract `.data.normalized_result` from both Evidence files and run
`rea compare-web-captures before.nr.json after.nr.json`. REA ignores elapsed time by
design, so compare **latency** by diffing the two `requests.md` files. Two back-to-back
captures of the same build compare `unchanged` (verified on the tail of this flow).

Notes for a faithful re-run: the narration and analyst-panel caches must be empty (a cached
panel hides the *run* button and the scenario's `s09-panel` click fails), so rebuild the
demo or delete `artifacts/{narration,panel}_cache.sqlite*` under the demo dir and restart
the server. Expect the run to take ~21 min while narration times out (finding #1).

## Isolation and scrubbing

- The browser only ever talked to `127.0.0.1:8025/8024`. Header **values** were not
  captured (names only), no response bodies, no request bodies; cookies and web storage were
  empty at the end. The scenario declares no secrets.
- The demo book is `scripts/capture_screenshots.py`'s seed (AAPL/MSFT/NVDA/ORCL/NFLX, three
  invented positions). The user's real `ops_state.sqlite` (watchlist, positions, alerts) was
  never opened by the demo; narration/panel caches were demo-local and started empty.
- `.env` is pre-empted: `STOCKSCAN_INTRINIO_KEY=""`, so the paid key was never used —
  `/api/live/quote` returns `null` and profiles/market caps come from copied caches. All
  saved files were swept for the `.env` secret values and the EDGAR contact address:
  no matches.
- Real inputs are read-only: price/fundamental parquet is symlinked, model heads and the
  paper store are copied. The demo has its own (empty) matrix cache because the real one
  is corrupt (see the Phase-1 report).

## Environment at capture time

Local Ollama 0.40.4, `gemma4:26b` warm (loaded with `OLLAMA_CONTEXT_LENGTH=262144`, set
machine-wide). Narration = "full" tier (gemma4:26b, uncapped, 600 s timeout, 1 retry);
panel/ask = "chat" tier (gemma4:26b, 500-token cap, reasoning off, 120 s timeout).

## LLM latency across every run of this flow (seconds, proxy-measured)

The saved capture is run D. Narration is the one call whose latency is not stable, so a
single run under-states it — this table is part of the baseline:

| run | narrate | panel memos bull/bear/risk/synthesis | ask |
|---|---|---|---|
| 1 · aborted (MCP client timeout), cold model | 1200.1 | — | — |
| A · full flow, stopped by a scenario selector bug at Markets | 1200.1 | 12.8, 9.4, 6.6, 9.8 (Σ 38.5) | 9.4 |
| C · full flow, request bodies on (not equality-eligible) | 1200.1 | 10.1, 8.9, 3.5, 8.5 (Σ 30.9) | 7.1 |
| **D · full flow — saved baseline** | **180.0** | 4.3, 10.1, 4.5, 10.9 (Σ 29.8) | 7.4 |

1200.1 s = two full-tier attempts each cut at the 600 s client timeout (Ollama logged
`500 | 10m0s` for 6 of the 7 full-tier calls that day; the 7th returned in 2m59s).
