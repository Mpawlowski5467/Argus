"""Production LLM telemetry — measure the honesty machinery instead of assuming it.

The bench scripts measure the ask path offline; nothing measured what users actually
see. Every shown grounded turn (ticker ask, book ask, explain-move, morning brief,
fresh narration) now logs one row to ops_state.sqlite: context hash + JSON, question,
answer, retry count, refusal, latency, token usage. The nightly judge job samples
recent non-refused turns and runs the advisory faithfulness judge over them, so
paraphrase-level drift in production gets measured nightly rather than assumed from
two bench runs. All local, like everything else.

Writes are fail-open: telemetry must never break or slow a user-facing answer —
a logging error is swallowed (the answer already exists; losing one row is nothing,
losing an answer is a bug).
"""

from __future__ import annotations

import hashlib
import json

_CONTEXT_CAP = 60_000     # a runaway context is truncated, not refused


def context_hash(context: dict) -> str:
    """Stable 12-hex fingerprint of a grounding context (canonical JSON). Ties a
    turn to the exact facts it answered from — same hash, same facts."""
    canon = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def record_turn(surface: str, context: dict, question: str, res: dict,
                latency_s: float, usage: dict | None = None,
                keep_context: bool = True) -> None:
    """Log one shown turn. Never raises."""
    try:
        from ..ops.state import OpsState

        ctx_json = json.dumps(context, default=str)[:_CONTEXT_CAP] if keep_context else None
        usage = usage or {}
        with OpsState() as st:
            st.log_llm_turn(
                surface=surface,
                context_hash=context_hash(context),
                question=(question or "")[:500],
                answer=(res.get("answer") or "")[:4000],
                attempts=int(res.get("attempts") or 0),
                refused=bool(res.get("refused")),
                latency_ms=int(latency_s * 1000),
                tokens_in=usage.get("prompt_tokens"),
                tokens_out=usage.get("completion_tokens"),
                context=ctx_json,
            )
    except Exception:
        pass


def judge_sample(state, llm, since: str, limit: int = 3) -> dict:
    """Run the advisory faithfulness judge over a sample of recent shown turns.

    Fail-open like the judge itself: a transport error or unparseable verdict counts
    as faithful (the deterministic numeral guard is the guarantee; this measures the
    paraphrase-level gap it can't see). Flagged turns raise ONE in-app alert per run —
    advisory, never high-severity, so a false flag can't train alert-blindness."""
    from .judge import judge_narration

    turns = state.unjudged_turns(since=since, limit=limit)
    judged, flagged = 0, []
    for t in turns:
        try:
            packet = json.loads(t["context"])
        except (TypeError, json.JSONDecodeError):
            state.set_turn_judgement(t["id"], [])
            continue
        verdict = judge_narration(t["answer"], packet, llm)
        state.set_turn_judgement(t["id"], verdict.get("issues") or [])
        judged += 1
        if not verdict.get("faithful", True):
            flagged.append({"id": t["id"], "surface": t["surface"],
                            "n_issues": len(verdict.get("issues") or [])})
    if flagged:
        state.add_alert(
            "judge_flag",
            f"faithfulness judge flagged {len(flagged)} of {judged} sampled LLM "
            f"turn(s) — review with: ops.py llm-turns",
            payload={"flagged": flagged})
    return {"sampled": len(turns), "judged": judged, "flagged": len(flagged)}
