"""Firewall / look-ahead auditor — guard the project's DNA continuously.

Two layers:

1. A DETERMINISTIC import scan (no LLM, no network): the signal/data core must never
   import the live-view/AI side. If ``model``/``backtest``/``panel``/``features``/… ever
   imports ``news``/``newsmem``/``narrate``/``assist``/``quote``/``tui``, that is a
   firewall breach — news or an LLM has a path into the score. This is exact and cheap,
   and is the check worth wiring into CI. Direction matters: the live side may import
   the core (to read data); the core may not import the live side.

2. An optional LLM review of a git diff against a firewall + point-in-time rubric, for
   the subtler leaks a grep can't see (using future data, a look-ahead join, a news
   value sneaking into a feature). Reuses the local model; grounded only in the diff.
"""

from __future__ import annotations

import ast
from pathlib import Path

# The live-view / AI side — never a dependency of the signal/data core. Everything
# here reads computed data or narrates it; none of it may flow INTO a score/backtest/
# panel/feature. (profile/marketcap/themes are the live-view metadata layer — included
# ahead of their merge so the auditor covers them the moment they land.)
FORBIDDEN = frozenset({
    "news", "newsmem", "narrate", "assist", "quote", "view", "web",
    "profile", "marketcap", "themes", "portfolio",
})
# Orchestration boundaries that LEGITIMATELY bridge core <-> live-view: serve builds
# the packet and narrates AFTER scoring; ops runs the monitor/news jobs; config is
# shared infra. Everything else under stockscan is treated as core and must import
# nothing from FORBIDDEN — so a new model head is protected automatically, no list edit.
BOUNDARY = frozenset({"serve", "ops", "config"})


def _imported_submodules(path: Path, pkg_parts: list[str]) -> set[str]:
    """Top-level ``stockscan`` submodules imported by one file (abs + relative)."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, OSError):
        return set()
    out: set[str] = set()

    def _record(dotted: str) -> None:
        parts = dotted.split(".")
        if parts and parts[0] == "stockscan" and len(parts) > 1:
            out.add(parts[1])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                _record(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                _record(node.module or "")
            else:
                # relative: climb `level-1` packages up from this file's package
                up = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                base = ".".join(up + ([node.module] if node.module else []))
                _record(base)
    return out


def firewall_scan(src_root: str | Path, forbidden=FORBIDDEN,
                  boundary=BOUNDARY) -> list[dict]:
    """Scan the signal/data core for any forbidden (live-view/AI) import.

    Denylist by design: EVERY top-level ``stockscan`` module/package is treated as core
    — and must import nothing from ``forbidden`` — except the live-view side itself
    (``forbidden``) and the sanctioned bridges (``boundary``). So a newly added model
    head is covered automatically. ``src_root`` is the package root (…/src/stockscan).
    Returns violation dicts ``{module, file, imports}``; empty means the firewall holds."""
    root = Path(src_root)
    forbidden, boundary = set(forbidden), set(boundary)
    violations: list[dict] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("")            # e.g. edgar/fsds
        top = rel.parts[0]                                      # top-level submodule name
        if top in forbidden or top in boundary or top.startswith("__"):
            continue                                            # live-side / bridge / dunder
        pkg_parts = ["stockscan", *rel.parts[:-1]]
        bad = sorted(_imported_submodules(path, pkg_parts) & forbidden)
        if bad:
            violations.append({"module": top,
                               "file": str(rel) + ".py", "imports": bad})
    return violations


FIREWALL_RUBRIC = (
    "You are auditing a code diff for a quant-equity project whose ONE inviolable rule "
    "is: news, LLM narration, and any live-view data must NEVER touch the scoring model, "
    "the backtest, the point-in-time panel, or a feature. Also flag look-ahead / "
    "point-in-time violations (using data not knowable at the decision date, a join that "
    "leaks the future, a label computed from same-day info). For EACH issue return an "
    "object {severity: 'critical'|'warn', file, line_hint, why}. If the diff is clean, "
    "return an empty issues list. Respond with ONLY JSON: {\"issues\": [ ... ]}. "
    "Report only real, defensible issues grounded in the diff — do not speculate."
)


def firewall_review_diff(diff_text: str, llm) -> dict:
    """LLM review of a git diff against the firewall + look-ahead rubric.

    Returns ``{issues: [...], raw}``. ``llm`` is a callable(system, user) -> str.
    Best-effort JSON parse; the deterministic :func:`firewall_scan` is the guarantee,
    this is the extra set of eyes."""
    import json
    import re

    try:
        raw = llm(FIREWALL_RUBRIC, "DIFF:\n" + diff_text)
    except Exception as exc:
        return {"issues": [], "raw": f"llm-error:{type(exc).__name__}: {exc}"}
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return {"issues": [], "raw": raw}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"issues": [], "raw": raw}
    issues = obj.get("issues", []) if isinstance(obj, dict) else []
    return {"issues": issues if isinstance(issues, list) else [], "raw": raw}
