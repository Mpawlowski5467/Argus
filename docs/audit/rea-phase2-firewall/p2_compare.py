"""Phase-2 runtime firewall check: compare the verdict/confidence chips across conditions.

For every condition's REA Evidence: the /api/ticker/{cik} response body (sha256 + the
verdict-bearing fields), the chip text actually rendered (accessibility snapshot after the
ticker settled), and which /api/* routes were answered vs blocked. Prints markdown.

    python3 -I p2_compare.py <dir-with-*.result.json> C0a C0b C1 C2 C3 C4
"""

import base64
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
CONDS = sys.argv[2:]
NAMES = {"320193": "AAPL", "1045810": "NVDA", "1083301": "WULF"}
STEP = {"AAPL": "aapl-settle", "NVDA": "nvda-settle", "WULF": "wulf-settle"}
CHIP = re.compile(
    r"\b(BUY|HOLD|AVOID|N/A) ((?:top-quintile|mid|bottom) signal \(\d+th pct\) — "
    r"(?:enters the long book|hold if owned, no new buy|long-only, so no position)|no score)"
    r"(?: (confidence \d+/100))?(?: (beat the median stock \d+% of the time OOS(?: \(n=[\d,]+\))?))?")
VERDICT_FIELDS = ("verdict", "score", "percentile", "decile")


def load(cond):
    nr = json.load(open(D / f"{cond}.result.json"))["data"]["normalized_result"]
    tx, bodies = {}, {}
    for it in nr["events"]["items"]:
        k = it.get("kind")
        if k == "response":
            tx[it["transaction_id"]] = (it["url"]["url"].split("8025", 1)[1], it["status"])
        elif k == "network-content" and it.get("phase") == "response":
            b = it.get("body") or {}
            if b.get("state") == "captured":
                bodies[it["transaction_id"]] = base64.b64decode(b["content"])
    ticker, routes = {}, collections.Counter()
    for tid, (path, status) in tx.items():
        if path.startswith("/api/"):
            routes[(path.split("?")[0].rstrip("0123456789").rstrip("/") or path, status)] += 1
        m = re.match(r"/api/ticker/(\d+)$", path)
        if m:
            raw = bodies.get(tid)
            ticker[NAMES[m.group(1)]] = {"status": status, "raw": raw,
                                         "sha": hashlib.sha256(raw).hexdigest()[:16] if raw else None,
                                         "json": json.loads(raw) if raw and status == 200 else None}
    chips, page = {}, {}
    steps = {s["step_id"]: s for s in nr["steps"]}
    for tk, sid in STEP.items():
        ax = ((steps.get(sid) or {}).get("artifacts", {}).get("accessibility") or {}).get("value") or {}
        text = ax.get("text", "")
        m = CHIP.search(text)
        chips[tk] = " · ".join(g for g in m.groups() if g) if m else None
        star = re.search(r"(★ watching|☆ watch)", text)
        err = re.search(r"- text: (unavailable|[^\n]*\b(?:503|blocked)[^\n]*)", text)
        page[tk] = {"star": star.group(1) if star else None,
                    "position": bool(re.search(r"(?i)your position|unrealized|cost basis", text)),
                    "err": err.group(1)[:60] if err else None}
    return {"ticker": ticker, "chips": chips, "page": page, "routes": routes,
            "complete": nr["completeness"]["status"]}


R = {c: load(c) for c in CONDS}
base = R[CONDS[0]]

print("## Chips as rendered (accessibility text after each ticker settled)\n")
print("| condition | " + " | ".join(NAMES.values()) + " |")
print("|---|" + "---|" * len(NAMES))
for c in CONDS:
    print(f"| {c} | " + " | ".join(f"{R[c]['chips'][t] or '— (no chip) ' + (R[c]['page'][t]['err'] or '')}"
                                    for t in NAMES.values()) + " |")

print("\n## /api/ticker body vs " + CONDS[0] + "\n")
print("| condition | ticker | status | sha256[:16] | body identical | verdict fields identical | confidence | drawdown flag | distress flag |")
print("|---|---|---|---|---|---|---|---|---|")
for c in CONDS:
    for t in NAMES.values():
        x, b = R[c]["ticker"].get(t), base["ticker"].get(t)
        if not x:
            print(f"| {c} | {t} | not requested | | | | | | |"); continue
        j, bj = x["json"], b["json"]
        same_body = x["raw"] == b["raw"]
        if j is None:
            print(f"| {c} | {t} | {x['status']} | {x['sha']} | {'yes' if same_body else 'no'} | n/a | | | |"); continue
        vf = all(j.get(k) == bj.get(k) for k in VERDICT_FIELDS)
        conf = (j.get("confidence") or {}).get("score")
        dd = (j.get("drawdown") or {}).get("flag")
        ds = (j.get("distress") or {}).get("flag")
        print(f"| {c} | {t} | {x['status']} | `{x['sha']}` | {'**yes**' if same_body else 'no'} | "
              f"{'**yes**' if vf else '**NO**'} | {conf} | {dd} | {ds} |")

print("\n## Book / watch state the page showed (proves the perturbation landed)\n")
print("| condition | " + " | ".join(NAMES.values()) + " |")
print("|---|" + "---|" * len(NAMES))
for c in CONDS:
    print(f"| {c} | " + " | ".join(f"{R[c]['page'][t]['star']} · position block: {'yes' if R[c]['page'][t]['position'] else 'no'}"
                                    for t in NAMES.values()) + " |")

print("\n## /api routes answered per condition (path family, status: count)\n")
for c in CONDS:
    rs = ", ".join(f"`{p}` {s}×{n}" for (p, s), n in sorted(R[c]["routes"].items()))
    print(f"- **{c}** ({R[c]['complete']}): {rs}")

# which top-level keys differ in the ticker body, per condition (excluding identical)
print("\n## Ticker-body keys that differ from " + CONDS[0] + "\n")
for c in CONDS[1:]:
    for t in NAMES.values():
        x, b = R[c]["ticker"].get(t), base["ticker"].get(t)
        if x and x["json"] and b["json"] and x["raw"] != b["raw"]:
            diff = sorted(k for k in set(x["json"]) | set(b["json"]) if x["json"].get(k) != b["json"].get(k))
            print(f"- {c} {t}: {diff}")
