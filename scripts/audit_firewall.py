"""Audit the firewall: the signal/data core must never import the live-view/AI side.

  uv run python scripts/audit_firewall.py                # deterministic import scan (fast)
  uv run python scripts/audit_firewall.py --diff         # + LLM review of `git diff HEAD`
  uv run python scripts/audit_firewall.py --diff main...HEAD   # + LLM review of a range

Exit code is 1 if the deterministic scan finds a breach — wire it into CI to keep news /
narration / any live-view path out of the score forever.
"""

import argparse
import subprocess

from stockscan.assist.audit import firewall_review_diff, firewall_scan
from stockscan.config import REPO_ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description="Firewall / look-ahead auditor.")
    ap.add_argument("--diff", nargs="?", const="HEAD", default=None, metavar="RANGE",
                    help="also LLM-review a git diff (default HEAD = working tree; or a range)")
    args = ap.parse_args()

    src = REPO_ROOT / "src" / "stockscan"
    viol = firewall_scan(src)
    if not viol:
        print("firewall import scan: CLEAN — the signal core imports nothing from the "
              "live-view/AI side.")
    else:
        print(f"firewall import scan: {len(viol)} BREACH(es)")
        for v in viol:
            print(f"  [BREACH] core module '{v['module']}' ({v['file']}) imports "
                  f"forbidden {v['imports']}")

    if args.diff is not None:
        rng = [] if args.diff == "HEAD" else args.diff.split()
        diff = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", *(rng or ["HEAD"])],
            capture_output=True, text=True).stdout
        if not diff.strip():
            print("\nLLM diff review: (empty diff — nothing to review)")
        else:
            from stockscan.narrate.llm import make_llm

            print("\nLLM diff review (subtler firewall / look-ahead leaks) ...")
            result = firewall_review_diff(diff, make_llm("full"))
            issues = result["issues"]
            if not issues:
                print("  no issues reported by the model review.")
            for it in issues:
                sev = str(it.get("severity", "?")).upper()
                print(f"  [{sev}] {it.get('file','?')}:{it.get('line_hint','?')} — "
                      f"{it.get('why','')}")

    return 1 if viol else 0


if __name__ == "__main__":
    raise SystemExit(main())
