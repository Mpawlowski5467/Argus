"""Sector scan: the ranked table renders instantly; narration is lazy for the top N.

  uv run python scripts/scan.py Finance --top 5 --no-llm
  uv run python scripts/scan.py all --top 8 --as-of 2026-06-30
  uv run python scripts/scan.py Manufacturing --model qwen3.6:27b-mlx --light phi4

The table is fully deterministic (frozen model over the PIT cross-section). The
top-N names then flow through the identical per-ticker analyze() path with
materiality-gated, cached, tiered narration (unchanged packet -> cache; minor
change -> light tier; material -> full tier).
"""

import argparse
import time

from stockscan.model import load_artifact
from stockscan.narrate.cache import NarrationCache, narrate_smart
from stockscan.narrate.llm import LocalLLM
from stockscan.serve import analyze, build_cross_section, load_serve_data


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic sector scan + lazy narration.")
    ap.add_argument("sector", help="SIC division (e.g. Finance, Manufacturing) or 'all'")
    ap.add_argument("--top", type=int, default=5, help="names to narrate (default 5)")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--no-llm", action="store_true", help="template narration only")
    ap.add_argument("--model", default=None, help="full-tier Ollama model tag")
    ap.add_argument("--light", default=None, help="light-tier model tag for minor changes")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("loading data + frozen artifact ...")
    data = load_serve_data()
    artifact = load_artifact()
    as_of = args.as_of or data.close.index[-1]

    cross = build_cross_section(data, as_of)
    cross = cross.reset_index(drop=True)
    cross["score"] = artifact.score(cross)
    cross["pct"] = cross["score"].rank(pct=True)
    view = cross if args.sector.lower() == "all" else cross[cross["sector"] == args.sector]
    if view.empty:
        print(f"no names in sector {args.sector!r}; known: {sorted(cross['sector'].dropna().unique())}")
        return 1
    view = view.sort_values("score", ascending=False)

    t_table = time.perf_counter() - t0
    print(f"\n== {args.sector}: {len(view)} names as of "
          f"{str(as_of)[:10]}  (table in {t_table:.1f}s) ==")
    print(f"{'rank':>4}  {'ticker':<14} {'name':<36} {'model pct':>9}  {'FY':>4}")
    import pandas as pd
    for i, (_, r) in enumerate(view.head(25).iterrows(), 1):
        fy = str(int(r["fy"])) if pd.notna(r["fy"]) else "—"
        print(f"{i:>4}  {str(r['ticker']):<14} {str(r['name'])[:36]:<36} "
              f"{r['pct']:>8.0%}  {fy:>4}")

    # ---- lazy narration for the top N through the identical analyze() path
    llm_full = llm_light = None
    if not args.no_llm:
        llm_full = LocalLLM(model=args.model) if args.model else LocalLLM()
        llm_light = LocalLLM(model=args.light) if args.light else None
    cache = NarrationCache()

    print(f"\n== narrating top {args.top} ==")
    times = []
    for _, r in view.head(args.top).iterrows():
        t1 = time.perf_counter()
        res = analyze(int(r["cik"]), as_of=as_of, data=data, artifact=artifact, llm=None)
        nar = narrate_smart(res["packet"], llm_full=llm_full, llm_light=llm_light,
                            cache=cache)
        dt = time.perf_counter() - t1
        times.append(dt)
        tag = f"{nar.get('tier', '?')}/{nar['source']}"
        print(f"\n--- {res['packet']['meta']['name']}  [{r['ticker']}]  "
              f"({tag}, {dt:.1f}s, grounded={nar['grounded']}) ---")
        print(nar["narrative"])

    total = time.perf_counter() - t0
    print(f"\nscan wall time: table {t_table:.1f}s + narration {sum(times):.1f}s "
          f"= {total:.1f}s total  (per-name avg {sum(times)/max(len(times),1):.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
