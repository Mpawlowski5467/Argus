"""The AI analyst panel — bull / bear / risk / synthesis memos over ONE grounded context.

This borrows exactly one idea from the TradingAgents project: a reader thinks better
with adversarial perspectives than with a single narration. Everything else about
that project is deliberately NOT here — no trade decision, no position sizing, no
n-round debate (two memos from one model over one context produce rhetorical
symmetry, not information; the bear gets ONE look at the bull's memo instead), and
no feedback of any kind into the signal. LLM output can never be used to evaluate or
tune the quant model — model weights know how history played out, so an "LLM
backtest" is contaminated by construction.

Honesty machinery (all pre-existing, applied per role): grounded_answer's numeral
guard + ISO-date rule + refusal path; the monthly-peer-rank-never-a-forecast rule
and coincided-never-caused news rule in every system prompt; an advisory judge pass
that SUPPRESSES a memo whose paraphrases drift from the context (shown as withheld,
never silently dropped); and the firewall — this module is display-only, imported by
the view/web layer, never by the signal core (enforced by the import scan).

FIREWALL: reads the widened chat context; writes only the panel cache. Never touches
score / percentile / paper / trade paths.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..config import ARTIFACTS_DIR
from .core import grounded_answer
from .judge import judge_narration

PANEL_CACHE_PATH = ARTIFACTS_DIR / "panel_cache.sqlite"

ROLES = ("bull", "bear", "risk", "synthesis")

WITHHELD = ("[withheld — this memo failed the faithfulness check against the "
            "underlying data and is not shown]")

# Appended to every role's system prompt — the project's non-negotiables, stated
# once so no role prompt can forget them. grounded_answer adds the ISO-date rule.
_COMMON_RULES = (
    "\nNON-NEGOTIABLE RULES:"
    "\n- The model signal is a MONTHLY CROSS-SECTIONAL PEER RANK among scored names."
    " It is never a return forecast, price target, or long-term view — refer to it"
    " only as a relative rank."
    "\n- News items are reported claims that COINCIDED with price moves or facts —"
    " never state that news caused anything, and never state a reported claim as"
    " established fact."
    "\n- No buy/sell/hold language, no advice, no predictions. You write commentary"
    " about evidence, not a recommendation."
    "\n- Use ONLY numbers that appear in the CONTEXT; where you have no number,"
    " describe direction in words. Never compute new figures."
    "\n- Refer to fundamentals by their PERCENTILE ranks (the whole numbers in the"
    " context) — never restate a raw ratio as a percentage or convert any value."
    "\n- 3-6 sentences. Plain prose, no headers, no bullet lists."
)

PANEL_SYSTEMS = {
    "bull": (
        "You are the BULL analyst on a research panel for a personal equity tool. "
        "From the CONTEXT alone, write the strongest HONEST case for what is going "
        "well at this company: the fundamentals reading in its favor, where it ranks "
        "among peers this month, and any favorable items in the news memory. Do not "
        "soften it with hedges the data doesn't force; do not overstate what a peer "
        "rank means. If little in the context favors the name, say so plainly — a "
        "thin bull case is a valid memo; never manufacture one." + _COMMON_RULES),
    "bear": (
        "You are the BEAR analyst on a research panel for a personal equity tool. "
        "From the CONTEXT alone, write the strongest HONEST case for what is going "
        "badly or is fragile at this company: the LEAST favorable signals (even for a "
        "strong name, some rank lower than others — name them and their direction in "
        "words), risk flags, data-quality caveats, unfavorable news items. If a BULL "
        "MEMO is provided in the question, end by rebutting its single weakest claim "
        "— using only the context's facts. If the context offers little against the "
        "name, say so plainly — a thin bear case is a valid memo; never manufacture "
        "one." + _COMMON_RULES),
    "risk": (
        "You are the RISK analyst on a research panel for a personal equity tool. "
        "From the CONTEXT alone, cover ONLY the risk reads: the distress flag, the "
        "drawdown flag, the confidence score and its hit-rate (say plainly what they "
        "do and don't mean), and the data-quality flags (filing staleness, liquidity, "
        "in-sample). These are display-only reads from firewalled models — say so. "
        "Do not re-argue the bull or bear case." + _COMMON_RULES),
    "synthesis": (
        "You are the SYNTHESIS analyst on a research panel for a personal equity "
        "tool. You are given the panel's memos in the question. State where the bull "
        "and bear ACTUALLY disagree about the same fact versus merely emphasize "
        "different facts; what evidence in future filings would settle each real "
        "disagreement; and what bounds the risk memo puts on both cases. You must "
        "NOT pick a side, score the debate, or lean the reader — end by reminding "
        "them this panel is commentary over one month's data, not part of the "
        "signal." + _COMMON_RULES),
}

_QUESTIONS = {
    "bull": "Write the bull memo.",
    "bear": "Write the bear memo.",
    "risk": "Write the risk memo.",
    "synthesis": "Write the synthesis memo.",
}


def role_question(role: str, prior: dict[str, str] | None = None) -> str:
    """The user-turn for a role: its task, plus the prior memos it is entitled to
    see (bear sees bull; synthesis sees all three). Prior memos ride in the QUESTION
    — they are generated prose, not grounding facts, and every numeral they carry
    already passed the guard against this same context."""
    prior = prior or {}
    parts = []
    if role == "bear" and prior.get("bull"):
        parts.append("BULL MEMO (rebut its weakest claim):\n" + prior["bull"])
    if role == "synthesis":
        for r in ("bull", "bear", "risk"):
            if prior.get(r):
                parts.append(f"{r.upper()} MEMO:\n{prior[r]}")
    parts.append(_QUESTIONS[role])
    return "\n\n".join(parts)


def panel_role(role: str, context: dict, llm, prior: dict[str, str] | None = None,
               judge_llm=None) -> dict:
    """Generate ONE role's memo: grounded_answer (numeral guard, retry, refusal) then
    an advisory judge pass that withholds a drifting memo instead of showing it.
    Returns the grounded_answer dict plus ``role``, ``judge_issues``, ``suppressed``."""
    if role not in ROLES:
        raise ValueError(f"unknown panel role: {role!r}")
    # panel memos get one retry more than chat: a memo is generated once per data
    # change (cached), so an extra roll against phrasing noise is cheap here and a
    # refusal is a whole missing panel section rather than one lost chat turn
    res = grounded_answer(context, role_question(role, prior), llm,
                          PANEL_SYSTEMS[role], max_retries=2)
    out = {**res, "role": role, "judge_issues": [], "suppressed": False}
    if judge_llm is not None and not res.get("refused"):
        verdict = judge_narration(res["answer"], context, judge_llm)
        out["judge_issues"] = verdict.get("issues") or []
        if not verdict.get("faithful", True):
            out["suppressed"] = True
            out["shown"] = WITHHELD
    if "shown" not in out:
        out["shown"] = out["answer"]
    return out


def build_panel(context: dict, llm, judge_llm=None) -> dict:
    """All four memos, sequentially (bear sees bull; synthesis sees all). A refused
    or suppressed memo is never fed forward as prior prose."""
    roles: dict[str, dict] = {}
    prior: dict[str, str] = {}
    for role in ROLES:
        r = panel_role(role, context, llm, prior=prior, judge_llm=judge_llm)
        roles[role] = r
        if not r.get("refused") and not r.get("suppressed"):
            prior[role] = r["answer"]
    return {"roles": roles, "generated_at": _utcnow()}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PanelCache:
    """Per-(cik, context-hash, role) memo cache. Fundamentals change quarterly, so a
    panel is generated once per data change, not per view; a NEW context hash for a
    cik evicts that cik's old rows (one live panel per name, never a stale mix)."""

    def __init__(self, path: Path = PANEL_CACHE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), timeout=10.0)
        self._db.execute(
            "create table if not exists panel ("
            " cik integer not null, context_hash text not null, role text not null,"
            " payload text not null, created text not null,"
            " primary key (cik, context_hash, role))")
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "PanelCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get(self, cik: int, context_hash: str) -> dict[str, dict]:
        rows = self._db.execute(
            "select role, payload from panel where cik = ? and context_hash = ?",
            (int(cik), context_hash)).fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}

    def put(self, cik: int, context_hash: str, role: str, payload: dict) -> None:
        # a fresh context hash retires the whole old panel for this name first
        self._db.execute(
            "delete from panel where cik = ? and context_hash != ?",
            (int(cik), context_hash))
        self._db.execute(
            "insert or replace into panel (cik, context_hash, role, payload, created) "
            "values (?,?,?,?,?)",
            (int(cik), context_hash, role, json.dumps(payload, default=str), _utcnow()))
        self._db.commit()
