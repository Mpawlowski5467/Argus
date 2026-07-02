"""Phase 0 gate: is the evaluation harness honest?

Builds the monthly cross-sectional panel (12-1 momentum feature, 63-day forward
excess-return label), computes rank IC, and runs two leakage smoke-tests that MUST
collapse to statistical zero: shuffled labels and a random feature. If either still
shows a significant IC, the harness is leaking and no downstream result can be trusted.

  uv run python scripts/run_gate.py
"""

import numpy as np

from stockscan.panel import build_panel, load_close_matrix
from stockscan.validation import ic_summary, rank_ic


def main() -> int:
    close = load_close_matrix()
    if close.empty:
        print("no prices found; run scripts/ingest_prices.py first")
        return 1
    panel = build_panel(close)
    if panel.empty:
        print("panel is empty (need >~1yr history for momentum + 63d forward for the label)")
        return 1

    n_dates = panel["date"].nunique()
    print(
        f"panel: {len(panel):,} rows  dates={n_dates}  "
        f"~{len(panel) // max(n_dates, 1)} names/date\n"
    )

    rng = np.random.default_rng(0)
    real = ic_summary(rank_ic(panel))

    shuffled = panel.copy()
    shuffled["label_excess"] = shuffled.groupby("date")["label_excess"].transform(
        lambda s: rng.permutation(s.to_numpy())
    )
    shuffled = ic_summary(rank_ic(shuffled))

    randfeat = panel.copy()
    randfeat["feature"] = rng.standard_normal(len(randfeat))
    randfeat = ic_summary(rank_ic(randfeat))

    def line(name, s):
        print(f"  {name:<20} mean_IC={s['mean_ic']:+.4f}  t_nw={s['t_nw']:+.2f}  (n={s['n']})")

    print("rank IC:")
    line("12-1 momentum", real)
    line("shuffled labels", shuffled)
    line("random feature", randfeat)

    honest = abs(shuffled["t_nw"]) < 2.0 and abs(randfeat["t_nw"]) < 2.0
    print()
    if honest:
        print("PHASE-0 GATE: PASS — harness is honest (both smoke-tests collapse to ~0).")
    else:
        print("PHASE-0 GATE: FAIL — a smoke-test stayed significant; hunt for leakage.")
    print(
        "Note: the momentum IC itself is not the gate — on a ~500-name, ~2yr slice it is\n"
        "just a sanity read. The real edge test is Phase 1 over the full history."
    )
    return 0 if honest else 2


if __name__ == "__main__":
    raise SystemExit(main())
