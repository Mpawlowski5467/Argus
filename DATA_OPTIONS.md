# Survivorship-Free Data Options (research, mid-2026)

> **OUTCOME (2026-07-01):** we went with **Intrinio** (the user's trial) — survivorship
> closed for real (7,003 dead + 4,132 active companies priced by security id), and the
> Phase-1 verdict was recomputed on it (see [RESULTS.md](RESULTS.md)). The ranking below is
> the pre-decision research, kept as the rationale of record — not current guidance.

Goal: replace free yfinance prices (which drop delisted names) so the Phase-1 IC becomes
trustworthy. **We already have clean point-in-time fundamentals (SEC EDGAR, segments-filtered)**,
so the thing we most need to buy is **survivorship-free PRICES** (delisted names' price history).

> Caveat: the research workflow's independent *verify* pass hit a rate limit, so these are
> single-pass findings. Prices are cited from vendor pages but **confirm the exact personal-license
> number at signup** — several are login-gated.

## The one structural finding

**No affordable vendor provides CRSP-style formal delisting *returns*** (the terminal
liquidation value at delisting). Every budget option instead **retains the delisted ticker's
price series up to its last trading day** — you model the final terminal tick yourself. We already
do exactly that (delisting ledger + imputation). So a budget feed with delisted prices removes the
*big* bias (missing living history) and leaves only the final-tick assumption we already handle.

Only **CRSP + Compustat (via WRDS)** gives true delisting returns + point-in-time fundamentals in
one place — and it needs a **university/employer affiliation** (no individual/alumni license).

## Ranked recommendation

| # | Option | ~Personal cost | Survivorship-free prices | PIT fundamentals | Verdict |
|---|---|---|---|---|---|
| **1** | **Tiingo — Power** | **$30/mo** ($300/yr) | Yes (delisted via `permaTicker`) | **Yes** (genuine `asReported`) | **Best documented value** — both capabilities, cheap, clean Python. No formal delisting return. |
| **2** | **Sharadar** (Nasdaq Data Link / QuantRocket) | gated (verify) | Yes (delisted retained) | **Yes** (SF1 ARQ/ARY as-first-reported) | The purpose-built quant option. Take it over #1 **if the gated personal price is ≲$50/mo**. |
| **3** | **Norgate — Platinum** | ~$30–35/mo (verify) | Yes (delisted incl.) | **No** (current-only) | Prices only → **pair with our EDGAR fundamentals**. Great if we just fix prices. |
| **4** | **CRSP + Compustat / WRDS** | ~$0 *if affiliated*, else N/A | Yes (+ real delisting returns) | Yes | Gold standard, but only with a university affiliation. |
| — | Intrinio | $150/mo | Yes | as-reported, XBRL-era (~2007+) | Solid but top-of-budget; fundamentals shallow. |
| — | EODHD (All-in-One) | $100/mo | Yes (12k delisted from 2000) | **Weak/no** true PIT | Prices ok, fundamentals not point-in-time. |
| — | FMP | $22–149/mo | DIY/weak | Weak (raw EDGAR) | Roughly what we already have free. |
| — | Polygon/Massive | ~$29/mo | delisted bars, **no** delist returns | No (re-parsed SEC XBRL) | Same fundamentals source we have free. |
| — | Databento | usage-based | reference/security-master only | **No fundamentals** | Great symbology, wrong product for us. |
| — | stockanalysis.com Pro | $10/mo | No (not a bulk dataset) | No (restated) | Fails both hard requirements. |

## If you pick one

- **On a budget, no affiliation → Tiingo Power ($30/mo).** It's the cheapest option that gives
  *both* delisted price history and genuine point-in-time fundamentals, with a clean Python API.
  Swap in its prices, keep (or cross-check) our EDGAR fundamentals, drop the circular imputation,
  and re-run — the IC becomes far more trustworthy for ~$30/mo.
- **Want the quant-native stack and the gated price is reasonable → Sharadar** (verify the
  personal-license figure at data.nasdaq.com or QuantRocket first).
- **Have any university affiliation → CRSP + Compustat via WRDS** (~$0 to you, the real gold
  standard with actual delisting returns).

## What changes in our pipeline once we have delisted prices

1. Point the `prices` module at the new feed (it's already behind a swappable interface).
2. Delisted names now have **real** living price history → the survivorship gap ≈ closed.
3. **Drop the delisting-return imputation** (the source of the ~4× circular IC inflation) — keep
   only a terminal-tick assumption for the actual delisting day.
4. Re-run the gate: the honest ~0.012–0.022 IC either firms up into a real (if modest) edge, or
   confirms there's no tradable signal — either way, a verdict you can finally trust.
