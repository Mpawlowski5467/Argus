"""Benchmark local narration models on ONE real packet -> pick the serving tiers.

Settles DESIGN.md §10's deferred decision (serving runtime + model) empirically on
the target M5 Pro: cold-load latency, warm latency, tokens/s, and whether the
model's first attempt survives the full grounding + citation validation.

  uv run python scripts/bench_llm.py [--company AAPL] [--models tag1,tag2,...]
"""

import argparse
import time

from stockscan.model import load_artifact
from stockscan.narrate.llm import LocalLLM
from stockscan.narrate.narrator import SYSTEM, parse_llm_json, validate_narration
from stockscan.serve import analyze, load_serve_data

DEFAULT_MODELS = "qwen3.6:27b-mlx,gemma4:26b,mistral-small3.1,gpt-oss:20b,phi4"


def main() -> int:
    ap = argparse.ArgumentParser(description="Local narration-model benchmark.")
    ap.add_argument("--company", default="AAPL")
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--runs", type=int, default=2, help="warm runs per model")
    args = ap.parse_args()

    print("building one real packet ...")
    data = load_serve_data()
    artifact = load_artifact()
    res = analyze(args.company, data=data, artifact=artifact, llm=None)
    import json
    user = json.dumps(res["packet"], indent=2, default=str)
    print(f"packet: {res['packet']['meta']['name']}  ({len(user):,} chars)\n")

    rows = []
    for tag in args.models.split(","):
        tag = tag.strip()
        llm = LocalLLM(model=tag, timeout=900.0)
        try:
            t0 = time.perf_counter()
            text = llm.complete(SYSTEM, user)
            cold = time.perf_counter() - t0

            warm_times, ok_count, out_tokens = [], 0, []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                text = llm.complete(SYSTEM, user)
                warm_times.append(time.perf_counter() - t0)
                out_tokens.append(llm.last_usage.get("completion_tokens") or 0)
                if not validate_narration(parse_llm_json(text), res["packet"]):
                    ok_count += 1
            warm = sum(warm_times) / len(warm_times)
            toks = sum(out_tokens) / len(out_tokens)
            rows.append((tag, cold, warm, toks, toks / warm if warm else 0,
                         f"{ok_count}/{args.runs}"))
            print(f"{tag:<22} cold {cold:6.1f}s  warm {warm:6.1f}s  "
                  f"{toks:5.0f} tok  {toks / warm:5.1f} tok/s  "
                  f"first-pass valid {ok_count}/{args.runs}")
        except Exception as exc:
            print(f"{tag:<22} FAILED: {type(exc).__name__}: {exc}")
            rows.append((tag, None, None, None, None, "error"))

    print("\n(warm = generation with the model already resident; cold includes load)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
