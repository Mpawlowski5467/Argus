"""Ask grounded questions about a ticker — read-only; every number traces to the packet.

  uv run python scripts/ask.py NFLX                       # interactive REPL
  uv run python scripts/ask.py NFLX "why is it ranked here?"   # one-shot
  uv run python scripts/ask.py NFLX --no-news            # fundamentals only

The answer is built from the SAME deterministic packet the narration uses (fundamentals,
the frozen model's signal + SHAP drivers, and recalled news). The model only frames the
numbers; a fabricated figure is caught and the assistant refuses rather than guesses.
Firewalled: the packet is assembled after scoring, so nothing here moves the signal.
"""

import argparse

from stockscan.assist.qa import answer_from_packet
from stockscan.model import load_artifact
from stockscan.narrate.llm import make_llm
from stockscan.narrate.packet import news_context
from stockscan.serve import analyze, load_serve_data, resolve_company


def _recall_news(cik: int) -> list:
    from stockscan.newsmem import NewsStore

    try:
        with NewsStore() as st:
            return st.context_for(int(cik))
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Grounded Q&A over a ticker.")
    ap.add_argument("ticker", help="ticker, TICKER~CIK, or CIK")
    ap.add_argument("question", nargs="*", help="one-shot question; omit for a REPL")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--no-news", action="store_true", help="fundamentals only")
    args = ap.parse_args()

    print("loading data + artifact ...", flush=True)
    data = load_serve_data()
    artifact = load_artifact()
    cik, column = resolve_company(args.ticker, data.ticker_map)
    news = None if args.no_news else news_context(_recall_news(cik))
    res = analyze(cik, as_of=args.as_of, data=data, artifact=artifact, llm=None, news=news)
    packet = res["packet"]
    llm = make_llm("chat")   # same capped client as the web ask surfaces

    m = packet["meta"]
    n_news = len(packet.get("context", {}).get("news", []))
    print(f"\n{m['name']} ({column}) — model {res['percentile']}th pct, decile "
          f"{res['decile']}/10 · {n_news} news items in context. "
          f"Ask a question (blank line to quit).\n")

    history: list = []

    def ask(q: str) -> None:
        r = answer_from_packet(packet, q, llm, history=history)
        tag = "  [refused: not in the data]" if r.get("refused") else ""
        print(f"\n{r['answer']}{tag}\n")
        if not r.get("refused"):
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": r["answer"]})

    if args.question:
        ask(" ".join(args.question))
        return 0
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        ask(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
