"""Per-ticker end-to-end analysis: parse -> compute -> score (frozen model) -> narrate.

  uv run python scripts/analyze.py AAPL                          # as of latest prices
  uv run python scripts/analyze.py BBBY~886158 --as-of 2022-09-30
  uv run python scripts/analyze.py AAPL JPM ETSY --no-llm --packet

Companies: ticker (AAPL), delisted column (BBBY~886158), or bare CIK (886158).
Everything is point-in-time at --as-of: latest 10-K with available_date <= as_of,
ranks vs that date's liquid cross-section, scored by the frozen artifact
(scripts/train_model.py). A delisted name runs through the identical path.
"""

import argparse
import json

from stockscan.narrate.llm import LocalLLM
from stockscan.narrate.narrator import _ord
from stockscan.serve import analyze, load_serve_data
from stockscan.model import load_artifact


def _print_result(r: dict) -> None:
    m = r["packet"]["meta"]
    f = r["flags"]
    print(f"\n=== {m['name']}  [{r['column'] or 'no price column'}  cik {r['cik']}] ===")
    print(f"as-of {m.get('as_of')}  |  FY{m['fiscal_year']} 10-K filed {f['filed_date']} "
          f"(usable {f['available_date']}, {f['staleness_days']}d old)  |  sector: {m['sector']}")
    mm = r["packet"]["model"]
    warn = "  [in-sample: as-of inside training window]" if f["in_sample"] else ""
    liq = "" if f["liquidity_pass"] else "  [below liquidity floor]"
    print(f"model signal: {_ord(mm['percentile'])} pct of {mm['n_names']} names "
          f"(decile {mm['decile']}/10, score {mm['score']:+.4f}; trained through "
          f"{mm['trained_through']}){warn}{liq}")
    print(f"\n{r['narrative']}")
    print(f"\n[narration: {r['source']}; grounded: {r['grounded']}]")


def main() -> int:
    ap = argparse.ArgumentParser(description="Point-in-time per-ticker analysis.")
    ap.add_argument("companies", nargs="+", help="ticker, TICKER~CIK column, or CIK")
    ap.add_argument("--as-of", default=None, help="analysis date (default: latest prices)")
    ap.add_argument("--no-llm", action="store_true", help="deterministic template only")
    ap.add_argument("--packet", action="store_true", help="also print the signal packet")
    args = ap.parse_args()

    print("loading data + frozen artifact ...")
    data = load_serve_data()
    artifact = load_artifact()
    llm = None if args.no_llm else LocalLLM()

    failures = 0
    for company in args.companies:
        query = int(company) if company.isdigit() else company
        try:
            r = analyze(query, as_of=args.as_of, data=data, artifact=artifact, llm=llm)
        except ValueError as exc:
            print(f"\n=== {company}: {exc} ===")
            failures += 1
            continue
        except Exception as exc:  # LLM endpoint unreachable -> deterministic template
            r = analyze(query, as_of=args.as_of, data=data, artifact=artifact, llm=None)
            r["source"] = f"template (llm unreachable: {type(exc).__name__})"
        _print_result(r)
        if args.packet:
            print(json.dumps(r["packet"], indent=2, default=str))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
