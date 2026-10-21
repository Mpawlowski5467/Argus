# REA Phase 2 — runtime firewall check of the verdict and confidence chips (2026-10-20)

Question: does anything from the book (positions, watchlist, alerts), the portfolio
scorecard, or the display-only risk flags reach the BUY/HOLD/AVOID chip or the confidence
chip on the ticker view?

Method: one REA scenario (`chips.json`: deep-link AAPL, then hash-navigate to NVDA and WULF,
accessibility snapshot + response bodies after each name settles) run against fresh
throwaway demo servers (Phase 1's `demo_setup.sh`) under six conditions, through
`fault_proxy.py`, which can answer chosen routes with 503:

| condition | what changed |
|---|---|
| C0a, C0b | demo as seeded, twice (determinism) |
| C1 | book rewritten through the app's own endpoints + two risk alerts (`c1_hook.sh`) |
| C2 | every `/api/*` except status/sectors/scan/ticker answered 503 by the proxy |
| C3 | distress + drawdown heads removed from the demo artifacts |
| C4 | `/api/ticker/*` answered 503 |

`p2_report.md` is the full output of `p2_compare.py` (chip text as rendered, ticker body
hashes and verdict fields per condition, which routes answered). Raw REA Evidence is not
kept here: it embeds the demo's third-party company-profile text.

Re-run: `DEMO=/tmp/argus-demo bash cond.sh C0a ""` (one condition per call; see the
headers of `cond.sh` and `c1_hook.sh` — the hook has the original demo path baked in).

`proposed_test_web_firewall.py` is the HTTP-level test proposed to lock this in
(`tests/test_web_firewall.py`). It passes on `rea-audit` today (4 tests, ~13 s) and was
checked against two deliberate mutants (ticker path reading the book; verdict ignoring the
percentile) — each is caught.
