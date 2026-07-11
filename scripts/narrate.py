"""Narrate one company's fundamentals with the local LLM (grounded so it invents no numbers).

  uv run python scripts/narrate.py AAPL            # local LLM if reachable, else template
  uv run python scripts/narrate.py AAPL --no-llm   # deterministic template only
  uv run python scripts/narrate.py AAPL --packet   # also print the signal packet

Local LLM: point STOCKSCAN_LLM_URL / STOCKSCAN_LLM_MODEL at Ollama or llama.cpp/MLX
(default http://localhost:11434/v1, qwen2.5:32b). `ollama pull qwen2.5:32b` on the M5 Pro.
"""

import argparse
import json

from stockscan.narrate.llm import make_llm
from stockscan.narrate.narrator import narrate


def main() -> int:
    ap = argparse.ArgumentParser(description="Grounded fundamental narration for one company.")
    ap.add_argument("company", help="ticker (e.g. AAPL) or CIK")
    ap.add_argument("--no-llm", action="store_true", help="deterministic template only")
    ap.add_argument("--packet", action="store_true", help="also print the signal packet")
    args = ap.parse_args()

    company = int(args.company) if args.company.isdigit() else args.company
    llm = None if args.no_llm else make_llm("full")
    try:
        result = narrate(company, llm=llm)
    except Exception as exc:  # LLM endpoint unreachable -> deterministic template
        result = narrate(company, llm=None)
        result["source"] = f"template (llm unreachable: {type(exc).__name__})"

    print(f"\n=== {result['packet']['meta']['name']} — narration [{result['source']}] ===\n")
    print(result["narrative"])
    print()
    if args.packet:
        print(json.dumps(result["packet"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
