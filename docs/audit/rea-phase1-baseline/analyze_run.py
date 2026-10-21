"""Join the REA scenario Evidence (event order, step, status, console) with the timing
proxy log (duration, bytes) and print the Phase-1 tables as markdown.

    python3 -I analyze_run.py scenario.evidence.json timing.jsonl > requests.md
"""

import collections
import json
import sys
from urllib.parse import urlsplit

ev_path, timing_path = sys.argv[1], sys.argv[2]
doc = json.load(open(ev_path))


def find_evidence(o):
    """The CLI envelope wraps the Evidence record; locate it by its normalized_result."""
    if isinstance(o, dict):
        if "normalized_result" in o and "operation" in o:
            return o
        for v in o.values():
            r = find_evidence(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_evidence(v)
            if r is not None:
                return r
    return None


ev = find_evidence(doc)
nr = ev["normalized_result"]
steps = {s["step_index"]: s for s in nr["steps"]}
items = nr["events"]["items"]

reqs = collections.OrderedDict()   # transaction_id -> record
console, page_errors = [], []
for it in items:
    k = it.get("kind")
    if k in ("request", "response", "request-finished", "request-failed", "request-unfinished"):
        tid = it["transaction_id"]
        u = it["url"]["url"]
        r = reqs.setdefault(tid, {"tid": tid, "step": it["step_index"], "method": it["method"],
                                  "url": u, "rtype": it.get("resource_type"), "status": None,
                                  "end": None, "failure": None, "seq": it["sequence"]})
        if k == "response":
            r["status"] = it.get("status")
        elif k != "request":
            r["end"] = k
            r["failure"] = it.get("failure") or it.get("reason")
    elif k == "console":
        console.append(it)
    elif k in ("page-error", "pageerror", "page-errors"):
        page_errors.append(it)
    elif "error" in (k or ""):
        page_errors.append(it)

timing = [json.loads(l) for l in open(timing_path)]
pool = collections.defaultdict(collections.deque)
for t in timing:
    pool[(t["method"], t["path"])].append(t)

rows = []
for r in reqs.values():
    sp = urlsplit(r["url"])
    path = sp.path + (("?" + sp.query) if sp.query else "")
    t = pool[(r["method"], path)].popleft() if pool[(r["method"], path)] else None
    rows.append({**r, "path": path, "t": t})

step_name = {i: s["step_id"] for i, s in steps.items()}


def fmt_ms(x):
    return "—" if x is None else (f"{x/1000:.1f} s" if x >= 10000 else f"{x:.0f} ms")


def fmt_b(x):
    if x is None:
        return "—"
    return f"{x/1024:.1f} KB" if x >= 1024 else f"{x} B"


print("## Steps\n")
print("| # | step | action | elapsed | status |")
print("|---|---|---|---|---|")
for i in sorted(steps):
    s = steps[i]
    print(f"| {i} | `{s['step_id']}` | {s['action']} | {fmt_ms(s.get('elapsed_ms'))} | {s['status']}"
          + (f" — {s['error'][:120]}" if s.get("error") else "") + " |")

print("\n## Every request (browser order)\n")
print("| # | step | method | route | status | duration | resp size | req size |")
print("|---|---|---|---|---|---|---|---|")
for n, r in enumerate(rows, 1):
    t = r["t"] or {}
    st = r["status"] if r["status"] is not None else (r["failure"] or r["end"] or "?")
    print(f"| {n} | `{step_name.get(r['step'], r['step'])}` | {r['method']} | `{r['path']}` | {st} | "
          f"{fmt_ms(t.get('duration_ms'))} | {fmt_b(t.get('resp_bytes'))} | {fmt_b(t.get('req_bytes') or None)} |")

print("\n## Slowest calls\n")
print("| route | step | duration | resp size |")
print("|---|---|---|---|")
for r in sorted([r for r in rows if r["t"]], key=lambda r: -r["t"]["duration_ms"])[:12]:
    print(f"| {r['method']} `{r['path']}` | `{step_name.get(r['step'])}` | {fmt_ms(r['t']['duration_ms'])} | {fmt_b(r['t']['resp_bytes'])} |")

print("\n## Repeated routes (same method+path more than once)\n")
cnt = collections.Counter((r["method"], r["path"]) for r in rows)
print("| route | count | steps |")
print("|---|---|---|")
for (m, p), c in cnt.most_common():
    if c > 1:
        st = sorted({step_name.get(r["step"]) for r in rows if (r["method"], r["path"]) == (m, p)})
        print(f"| {m} `{p}` | {c} | {', '.join(st)} |")

print("\n## Console / page errors\n")
for c in console:
    print(f"- console.{c.get('level')} (step `{step_name.get(c['step_index'])}`): {c.get('text','')[:200]} "
          f"— {c.get('url',{}).get('url') if isinstance(c.get('url'), dict) else c.get('url')}")
for e in page_errors:
    print(f"- page error (step `{step_name.get(e.get('step_index'))}`): {json.dumps(e)[:300]}")
if not console and not page_errors:
    print("- none")

unmatched = [t for q in pool.values() for t in q]
print(f"\n_proxy rows not matched to a browser request: {len(unmatched)}_")
for t in unmatched:
    print(f"- {t['method']} `{t['path']}` {t['status']} {fmt_ms(t['duration_ms'])} error={t['error']}")
tot = sum((r["t"] or {}).get("resp_bytes") or 0 for r in rows)
print(f"\n_total: {len(rows)} requests, {fmt_b(tot)} response bytes; scenario duration {fmt_ms(nr.get('duration_ms'))}_")
