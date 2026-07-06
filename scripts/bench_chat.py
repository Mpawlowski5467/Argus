"""Benchmark chat-tier models on the REAL grounded ask path → pick LLM_CHAT_MODEL.

Settles the hardening round's deferred decision empirically on the target machine:
can a small model (phi4) hold the interactive ask surfaces, or does chat stay on
the full narration model? Contexts are real (packet + display reads + recalled
news), built exactly as ``ArgusData.ask`` builds them; questions are the ticker
page's suggestion chips. Measured per model: per-turn wall time, first-pass
grounding (the deterministic guard), refusals, and — because grounding checks
numerals, not meaning — a FULL-model faithfulness judge over every shown answer
(assist.judge). Chat safety does not depend on the model (the guard refuses over
fabricating no matter what); this measures how much polish and latency each model
trades.

  uv run python scripts/bench_chat.py [--companies AAPL,MSFT,NVDA]
      [--models phi4,gemma4:26b] [--judge-model gemma4:26b] [--max-tokens 500]
"""

from __future__ import annotations

import argparse
import time

from stockscan.assist.judge import judge_narration
from stockscan.assist.qa import CHAT_SYSTEM
from stockscan.assist.core import grounded_answer
from stockscan.config import LLM_CHAT_MAX_TOKENS
from stockscan.narrate.llm import LocalLLM
from stockscan.view.data import ArgusData

QUESTIONS = [
    "why is it ranked here?",
    "what are the risk flags?",
    "what does the news say?",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Chat-tier model benchmark (real ask path).")
    ap.add_argument("--companies", default="AAPL,MSFT,NVDA")
    ap.add_argument("--models", default="phi4,gemma4:26b")
    ap.add_argument("--judge-model", default="gemma4:26b")
    ap.add_argument("--max-tokens", type=int, default=LLM_CHAT_MAX_TOKENS)
    args = ap.parse_args()

    print("loading the facade (data + scored cross-section) ...")
    ad = ArgusData.load()
    contexts: dict[str, dict] = {}
    for name in args.companies.split(","):
        name = name.strip()
        hit = ad.resolve(name)
        if not hit:
            print(f"  {name}: not resolvable — skipped")
            continue
        # the REAL ask context, by construction — the same helper /ask calls
        contexts[name] = ad.chat_context(hit["cik"])
        print(f"  {name}: context ready (cik {hit['cik']})")

    models = [m.strip() for m in args.models.split(",")]
    # all turns for one model before the next — the Ollama server swaps models
    # slowly, so ordering by model measures the model, not the reloads
    turns: list[dict] = []
    for tag in models:
        llm = LocalLLM(model=tag, timeout=300.0, max_tokens=args.max_tokens,
                       reasoning_effort="none")
        llm("You are a warmup.", "Say OK.")            # load the model off the clock
        for name, ctx in contexts.items():
            for q in QUESTIONS:
                t0 = time.perf_counter()
                r = grounded_answer(ctx, q, llm, CHAT_SYSTEM, max_retries=1)
                secs = time.perf_counter() - t0
                turns.append({"model": tag, "company": name, "q": q, "secs": secs,
                              "tokens": llm.last_usage.get("completion_tokens"),
                              **r})
                flag = ("REFUSED" if r["refused"]
                        else "ok" if r["attempts"] == 1 else f"retry x{r['attempts'] - 1}")
                print(f"{tag:<12} {name:<6} {q:<28} {secs:6.1f}s  {flag}")

    print(f"\njudging shown answers with {args.judge_model} ...")
    judge = LocalLLM(model=args.judge_model, timeout=600.0)
    for t in turns:
        if t["refused"]:
            t["faithful"] = None                       # refusal shows no claims to judge
            continue
        t["faithful"] = judge_narration(
            t["answer"], contexts[t["company"]], judge)["faithful"]

    print(f"\n{'model':<12} {'mean s':>7} {'p_max s':>8} {'1st-pass':>9} "
          f"{'refused':>8} {'faithful':>9}")
    for tag in models:
        mine = [t for t in turns if t["model"] == tag]
        if not mine:
            continue
        shown = [t for t in mine if not t["refused"]]
        first = sum(1 for t in mine if not t["refused"] and t["attempts"] == 1)
        faith = sum(1 for t in shown if t["faithful"])
        print(f"{tag:<12} {sum(t['secs'] for t in mine) / len(mine):7.1f} "
              f"{max(t['secs'] for t in mine):8.1f} "
              f"{first:>6}/{len(mine)} {sum(t['refused'] for t in mine):>8} "
              f"{faith:>6}/{len(shown)}")
    print("\nRead: chat honesty is guard-enforced either way; promote the small "
          "model only if refusals and judge-faithfulness hold up, for the speed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
