"""Faithfulness eval (Phase-4 gate): N real tickers through the full serve+narrate path.

Measures what the gate demands (DESIGN.md §8): fabricated numbers in FINAL output
(must be ~0 — the guard enforces it; this verifies), full citation traceability,
and the honest costs of the guard: raw first-pass fabrication rate, retry rate,
template-fallback rate, latency distribution, projected sector-scan window.

  uv run python scripts/eval_narration.py --n 40 --model qwen3.6:27b-mlx [--seed 7]
"""

import argparse
import time

import numpy as np

from stockscan.model import load_artifact
from stockscan.narrate.ground import check_grounding
from stockscan.narrate.llm import LocalLLM
from stockscan.narrate.narrator import validate_narration
from stockscan.serve import analyze, build_cross_section, load_serve_data


def main() -> int:
    ap = argparse.ArgumentParser(description="Narration faithfulness eval.")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default=None, help="Ollama model tag (default: config)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--as-of", default=None)
    args = ap.parse_args()

    print("loading data + artifact ...")
    data = load_serve_data()
    artifact = load_artifact()
    as_of = args.as_of or data.close.index[-1]
    cross = build_cross_section(data, as_of)
    pool = cross[cross["liquidity_pass"]]["cik"].astype(int).tolist()
    rng = np.random.default_rng(args.seed)
    sample = [int(c) for c in rng.choice(pool, size=min(args.n, len(pool)), replace=False)]
    llm = LocalLLM(model=args.model) if args.model else LocalLLM()
    print(f"evaluating {len(sample)} names as of {str(as_of)[:10]} "
          f"with model {llm.model}\n")

    stats = {"first_pass_ok": 0, "retried_ok": 0, "fallback": 0,
             "final_ungrounded": 0, "citation_fail": 0}
    lat, first_violations = [], []
    for i, cik in enumerate(sample, 1):
        t0 = time.perf_counter()
        try:
            r = analyze(cik, as_of=as_of, data=data, artifact=artifact, llm=llm)
        except Exception as exc:  # one bad name must never kill the eval
            print(f"  [{i:>3}] cik {cik}: skipped ({type(exc).__name__}: {exc})")
            continue
        dt = time.perf_counter() - t0
        lat.append(dt)

        # gate condition 1: zero fabricated numbers in the FINAL narration
        if check_grounding(r["narrative"], r["packet"]):
            stats["final_ungrounded"] += 1
        # gate condition 2: full citation traceability of the FINAL output
        if validate_narration({"reasoning": "", "summary": r["narrative"],
                               "citations": r["citations"]}, r["packet"]):
            stats["citation_fail"] += 1

        if r["source"] == "llm" and r["first_pass_ok"]:
            stats["first_pass_ok"] += 1
        elif r["source"] == "llm":
            stats["retried_ok"] += 1
        else:
            stats["fallback"] += 1
        vlog = r.get("violation_log") or (
            [r["narration_violations"]] if r.get("narration_violations") else [])
        if vlog and vlog[0]:
            first_violations.append((r["packet"]["meta"]["ticker"], vlog[0][:4]))
        print(f"  [{i:>3}] {str(r['packet']['meta']['ticker']):<14} "
              f"{r['source']:<18} attempts={r['attempts']}  {dt:5.1f}s  "
              f"grounded={r['grounded']}")

    n = len(lat)
    if not n:
        print("nothing evaluated")
        return 1
    lat_a = np.asarray(lat)
    print(f"\n== faithfulness ({n} names, model {llm.model}) ==")
    print(f"first-pass valid:        {stats['first_pass_ok']}/{n}"
          f"  (raw fabrication/contract-violation rate "
          f"{1 - stats['first_pass_ok'] / n:.0%})")
    print(f"passed on retry:         {stats['retried_ok']}/{n}")
    print(f"template fallback:       {stats['fallback']}/{n}")
    print(f"latency: mean {lat_a.mean():.1f}s  p50 {np.percentile(lat_a, 50):.1f}s  "
          f"p90 {np.percentile(lat_a, 90):.1f}s")
    print(f"projected COLD top-10 narration backfill: {lat_a.mean() * 10 / 60:.1f} min "
          f"(lazy/async per DESIGN §7; cached + materiality-gated on re-scans)")
    if first_violations:
        print("\nsample first-pass violations (guard caught, reader never saw):")
        for tick, v in first_violations[:8]:
            print(f"  {tick}: {v}")

    print("\n== PHASE-4 GATE ==")
    checks = {
        "zero fabricated numbers in final output": stats["final_ungrounded"] == 0,
        "full citation traceability in final output": stats["citation_fail"] == 0,
        # the table is instant and narration backfills lazily (DESIGN §7); the cold
        # full-tier window just has to be a coffee break, not an overnight job
        "scan window acceptable (cold top-10 backfill < 30 min)": lat_a.mean() * 10 < 1800,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nGATE: {'PASS' if all(checks.values()) else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
