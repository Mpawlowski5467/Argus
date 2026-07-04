"""LLM faithfulness judge — the paraphrase-level check the deterministic guard can't do.

The grounding guard proves every NUMERAL traces to the packet and every EXPLICIT
citation agrees in direction. What it can't see (Phase-4 documented this as the deferred
"LLM judge" gap): a fundamental referenced by a SYNONYM in the wrong direction with no
citation, or a news theme stated as established fact rather than reported. This judge
reads the packet + the shown narrative and flags exactly those semantic misses. It is a
QA instrument over narration output — never in the serve path, never near the score.
"""

from __future__ import annotations

import json
import re

JUDGE_SYSTEM = (
    "You are a STRICT faithfulness judge for equity-fundamentals narration. You are given "
    "a PACKET (the ONLY facts the narrator was allowed to use) and a NARRATIVE. Find every "
    "claim in the narrative that the packet does NOT support:\n"
    "- a number or magnitude not present in the packet;\n"
    "- a signal or model driver described in the WRONG direction — a weakness called a "
    "strength or vice versa — including when it is referenced by a synonym or paraphrase "
    "rather than its exact label (e.g. 'profitability' for return on assets);\n"
    "- a news item stated as established FACT instead of a reported claim, or attributed "
    "to the wrong source/company;\n"
    "- any buy/sell recommendation, price target, or return prediction.\n"
    "For each issue return {\"type\": \"<one of: number|direction|news|advice>\", "
    "\"quote\": \"<the offending span>\", \"why\": \"<one line>\"}. If the narrative is "
    "fully faithful to the packet, return an empty list. Respond with ONLY JSON: "
    "{\"issues\": [ ... ]}. Judge only what the text actually says; do not invent issues."
)


def judge_narration(narrative: str, packet: dict, llm) -> dict:
    """Judge one narrative against its packet. Returns ``{issues, faithful, raw}``.

    ``llm`` is a callable(system, user) -> str. Best-effort JSON parse; a parse failure
    or transport error yields no issues (fail-open — the judge is advisory, the
    deterministic guard is the guarantee)."""
    user = ("PACKET:\n" + json.dumps(packet, indent=2, default=str, sort_keys=True)
            + "\n\nNARRATIVE:\n" + str(narrative))
    try:
        raw = llm(JUDGE_SYSTEM, user)
    except Exception as exc:
        return {"issues": [], "faithful": True, "raw": f"llm-error:{type(exc).__name__}"}
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return {"issues": [], "faithful": True, "raw": raw}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"issues": [], "faithful": True, "raw": raw}
    issues = obj.get("issues", []) if isinstance(obj, dict) else []
    issues = [i for i in issues if isinstance(i, dict)] if isinstance(issues, list) else []
    return {"issues": issues, "faithful": not issues, "raw": raw}
