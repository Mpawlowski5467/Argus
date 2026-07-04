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

from stockscan.assist.judge import judge_narration
from stockscan.model import load_artifact
from stockscan.narrate.ground import check_grounding
from stockscan.narrate.llm import LocalLLM
from stockscan.narrate.narrator import news_ids, validate_narration
from stockscan.narrate.packet import _strip_numbers
from stockscan.news import company_news
from stockscan.serve import analyze, build_cross_section, load_serve_data


def _live_news_context(ticker: str, limit: int) -> list[dict]:
    """Number-free news takeaways from LIVE Intrinio headlines (A-gate stand-in for
    Part B's LLM extraction). Titles are number-stripped into takeaways, so a headline
    like 'Q3 EPS beats by $0.12' becomes number-free — an ADVERSARIAL news context:
    if the guard were porous, those stripped figures would slip back in via narration."""
    arts = company_news(ticker, limit=limit) if ticker else []
    out = []
    for a in arts:
        if not a.get("id"):
            continue
        out.append({
            "id": a["id"], "date": a.get("date") or "", "source": a.get("source") or "",
            "event_type": "other", "takeaway": _strip_numbers(a.get("title") or ""),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Narration faithfulness eval.")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default=None, help="Ollama model tag (default: config)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--news", type=int, default=0, metavar="K",
                    help="attach up to K live Intrinio headlines (number-free) as "
                         "narration context — the news-hardening acceptance gate")
    ap.add_argument("--judge", action="store_true",
                    help="also run the LLM faithfulness judge on each final narration "
                         "(catches the paraphrase-level misses the guard can't)")
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
             "final_ungrounded": 0, "citation_fail": 0, "with_news": 0, "judge_flagged": 0}
    lat, first_violations, judge_notes = [], [], []
    for i, cik in enumerate(sample, 1):
        news = _live_news_context(data.ticker_map.get(cik), args.news) if args.news else None
        if news:
            stats["with_news"] += 1
        t0 = time.perf_counter()
        try:
            r = analyze(cik, as_of=as_of, data=data, artifact=artifact, llm=llm, news=news)
        except Exception as exc:  # one bad name must never kill the eval
            print(f"  [{i:>3}] cik {cik}: skipped ({type(exc).__name__}: {exc})")
            continue
        dt = time.perf_counter() - t0
        lat.append(dt)
        nids = news_ids(r["packet"])
        cited_news = sum(1 for c in r["citations"]
                         if isinstance(c, dict) and c.get("id") in nids)

        if args.judge and r["source"] == "llm":
            jr = judge_narration(r["narrative"], r["packet"], llm)
            if not jr["faithful"]:
                stats["judge_flagged"] += 1
                judge_notes.append((r["packet"]["meta"].get("ticker"), jr["issues"][:2]))

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
        news_note = f"  news={len(news)}→cited={cited_news}" if news else ""
        print(f"  [{i:>3}] {str(r['packet']['meta']['ticker']):<14} "
              f"{r['source']:<18} attempts={r['attempts']}  {dt:5.1f}s  "
              f"grounded={r['grounded']}{news_note}")

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
    if args.news:
        print(f"names with news context: {stats['with_news']}/{n} "
              f"(fabrication gate must hold WITH news present)")
    if args.judge:
        print(f"LLM judge flagged:       {stats['judge_flagged']}/{n} "
              f"(paraphrase-level misses beyond the deterministic guard)")
        for tick, issues in judge_notes[:6]:
            print(f"  {tick}: {[i.get('type') + ':' + i.get('quote','')[:40] for i in issues]}")
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
