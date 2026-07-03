"""Narrate a company's signals with a local LLM, grounded so it invents nothing.

The Phase-4 contract (DESIGN.md §7): the LLM answers in JSON — a free-text
``reasoning`` field FIRST, then the constrained fields: a ``summary`` (the shown
narration) and ``citations`` naming every signal/driver referenced, each with a
direction. A deterministic post-validator then enforces three things:

1. every numeral in the summary AND the reasoning traces to the packet (the
   grounding guard — the narrator never does arithmetic);
2. every citation id exists in the packet;
3. every citation's direction agrees with the packet's own sign — a signal's
   effective percentile, a driver's SHAP contribution. The LLM cannot call a
   weakness a strength.

Violations -> bounded retry -> deterministic template fallback. The verdict/score
is never the LLM's; it only frames pre-computed numbers.
"""

from __future__ import annotations

import json
import re

from .ground import check_grounding
from .packet import build_packet

SYSTEM = (
    "You are a careful equity fundamentals analyst. You are given a JSON packet of "
    "already-computed signals for one company (values, sector percentiles, YoY changes, "
    "and the frozen model's signal with its top drivers).\n"
    "Respond with ONLY a JSON object, no code fences, in exactly this shape:\n"
    '{"reasoning": "<your free-text working, any length>",\n'
    ' "summary": "<a concise 120-180 word plain-English read of the fundamental profile '
    'and model signal>",\n'
    ' "citations": [{"id": "<signal or driver id from the packet>", '
    '"direction": "supports" | "detracts"}]}\n'
    "HARD RULES:\n"
    "- Use ONLY numbers that appear in the packet — in the summary AND the reasoning. "
    "Never invent, estimate, round differently, or compute new figures (no arithmetic, "
    "no counts). If a number you want is not in the packet, describe it in words.\n"
    "- Every signal or model driver you reference in the summary must appear in "
    "citations. COPY the direction from the packet — a signal's 'read' field and a "
    "driver's 'direction' field ARE the citation direction; never derive your own "
    "(e.g. shrinking assets can be a plus — the packet already decided). Use the "
    "exact 'id' field (model drivers are namespaced like 'driver:roa'); the same "
    "fundamental may legitimately carry different directions as signal vs driver. "
    "Any signal or driver you mention by name in the summary MUST have a citation — "
    "an uncited mention is rejected.\n"
    "- Respect each signal's 'direction' field: a HIGH percentile on a lower-is-better "
    "signal (leverage, accruals, asset growth) is a weakness, not a strength.\n"
    "- The model section is a relative rank from a frozen statistical model — describe "
    "it as such, never as a guaranteed return. The composite is a peer screen, not a "
    "forecast.\n"
    "- No buy/sell advice, price targets, or predictions. Analysis only."
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _ord(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _effective_pct(s: dict) -> float:
    """Goodness of a signal: its percentile, flipped for lower-is-better signals.

    A 98th-percentile leverage is a WEAKNESS (most-levered name in the sector),
    not a strength -- strongest/weakest ordering must respect the direction."""
    return s["pct_rank"] if s["direction"] == "higher-is-better" else 100 - s["pct_rank"]


def expected_directions(packet: dict) -> dict[str, str]:
    """The packet's own verdict per citable id — what a citation must agree with."""
    exp: dict[str, str] = {}
    for s in packet.get("signals", []):
        if s.get("pct_rank") is not None:
            # the packet's own 'read' is the single source of truth when present
            exp[s["id"]] = s.get("read") or (
                "supports" if _effective_pct(s) >= 50 else "detracts")
    for d in packet.get("model", {}).get("drivers", []):
        exp[d["id"]] = d["direction"]  # ids arrive namespaced ("driver:roa")
    comp = packet.get("composite", {}).get("percentile")
    if comp is not None:
        exp["composite"] = "supports" if comp >= 50 else "detracts"
    model_pct = packet.get("model", {}).get("percentile")
    if model_pct is not None:
        exp["model"] = "supports" if model_pct >= 50 else "detracts"
    return exp


def flexible_ids(packet: dict, band: float = 5.0) -> set:
    """Ids near the median (45-55 effective pct): either direction is defensible —
    forcing 'supports' at a rounded 50th percentile would reject honest narration."""
    flex: set = set()
    for s in packet.get("signals", []):
        if s.get("pct_rank") is not None and abs(_effective_pct(s) - 50) <= band:
            flex.add(s["id"])
    comp = packet.get("composite", {}).get("percentile")
    if comp is not None and abs(comp - 50) <= band:
        flex.add("composite")
    model_pct = packet.get("model", {}).get("percentile")
    if model_pct is not None and abs(model_pct - 50) <= band:
        flex.add("model")
    return flex


def _mention_map(packet: dict) -> dict[str, set]:
    """Signal/driver display labels -> the citation ids that cover a mention.

    The same fundamental can appear as a signal (textbook direction) AND a model
    driver (learned direction); a mention is covered if EITHER is cited."""
    m: dict[str, set] = {}
    for s in packet.get("signals", []):
        if s.get("pct_rank") is not None:
            m.setdefault(s["label"].lower(), set()).add(s["id"])
    for d in packet.get("model", {}).get("drivers", []):
        m.setdefault(d["label"].lower(), set()).add(d["id"])
    return m


def parse_llm_json(text: str) -> dict | None:
    """Parse the LLM's JSON reply (tolerating code fences / prose around it).

    Extracts the first brace-BALANCED object (string-aware) — slicing first-{ to
    last-} would let any brace in surrounding prose corrupt the parse and falsely
    reject a valid reply. Objects that parse but lack a summary are skipped so a
    small example object in a preamble can't shadow the real one."""
    cleaned = _FENCE.sub("", text.strip())
    pos = 0
    while (start := cleaned.find("{", pos)) >= 0:
        depth, in_str, esc = 0, False, False
        for j in range(start, len(cleaned)):
            c = cleaned[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:j + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "summary" in obj:
                        return obj
                    break
        pos = start + 1
    return None


def validate_narration(parsed: dict | None, packet: dict) -> list:
    """Deterministic post-validation. Returns a list of violations (empty = clean)."""
    if parsed is None:
        return ["unparseable-json"]
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return ["missing-summary"]
    violations: list = []
    violations += check_grounding(summary, packet)
    violations += check_grounding(str(parsed.get("reasoning", "")), packet)

    exp = expected_directions(packet)
    flex = flexible_ids(packet)
    cits = parsed.get("citations", [])
    if not isinstance(cits, list):
        return violations + ["citations-not-a-list"]
    cited_ids: set = set()
    for c in cits:
        if not isinstance(c, dict) or "id" not in c or "direction" not in c:
            violations.append(f"malformed-citation:{c!r}")
            continue
        cid, cdir = str(c["id"]), str(c["direction"])
        cited_ids.add(cid)
        if cid not in exp:
            violations.append(f"unknown-citation-id:{cid}")
        elif cdir not in ("supports", "detracts"):
            violations.append(f"bad-direction:{cid}:{cdir}")
        elif cdir != exp[cid] and cid not in flex:
            violations.append(f"direction-disagrees:{cid}:{cdir}!={exp[cid]}")

    # the direction guard must not be opt-out: a signal/driver MENTIONED by its
    # packet label in the shown summary must be cited (under either of its ids),
    # otherwise a wrong-direction claim could simply omit its citation
    low = summary.lower()
    for label, ids in _mention_map(packet).items():
        if label in low and not (ids & cited_ids):
            violations.append(f"uncited-mention:{sorted(ids)[0]}")
    return violations


def _strong_weak(packet: dict) -> tuple[list, list]:
    """Top-3 and bottom-3 signals by direction-aware goodness, never overlapping."""
    ranked = sorted(
        [s for s in packet["signals"] if s.get("pct_rank") is not None],
        key=_effective_pct,
        reverse=True,
    )
    return ranked[:3], ranked[3:][-3:]


def _fmt_val(v) -> str:
    # junk fundamentals (near-zero denominators) can be astronomically large;
    # scientific notation would emit numerals (mantissa, exponent) that trace to
    # nothing in the packet, so large values render in plain digits
    return f"{v:,.0f}" if isinstance(v, float) and abs(v) >= 1e5 else str(v)


def _template(packet: dict) -> str:
    m = packet["meta"]
    strong, weak = _strong_weak(packet)
    def fmt(s):
        note = "" if s["direction"] == "higher-is-better" else "; lower is better"
        return f"{s['label']} {_fmt_val(s['value'])}{s['unit']} ({_ord(s['pct_rank'])} pct{note})"
    strengths = "; ".join(fmt(s) for s in strong)
    weaker = "; ".join(fmt(s) for s in weak)
    comp = packet["composite"]["percentile"]
    comp_txt = f"composite quality {_ord(comp)} percentile vs sector" if comp is not None else "composite n/a"
    model_txt = ""
    if packet.get("model"):
        mm = packet["model"]
        model_txt = (
            f" Frozen-model signal (as of {mm['as_of']}, model trained through "
            f"{mm['trained_through']}): {_ord(mm['percentile'])} percentile of the "
            f"{mm['n_names']}-name cross-section (decile {mm['decile']}) — a relative "
            f"rank, not a return forecast."
        )
        drivers = mm.get("drivers", [])
        helps = [d["label"] for d in drivers if d["direction"] == "supports"]
        hurts = [d["label"] for d in drivers if d["direction"] == "detracts"]
        if helps or hurts:
            parts = []
            if helps:
                parts.append("supported by " + ", ".join(helps))
            if hurts:
                parts.append("held back by " + ", ".join(hurts))
            model_txt += " Model drivers: " + "; ".join(parts) + "."
    weak_txt = f" Weakest: {weaker}." if weaker else ""
    return (
        f"{m['name']} (FY{m['fiscal_year']}, {m['sector']}): {comp_txt}. "
        f"Strongest: {strengths}.{weak_txt}{model_txt} {packet['disclaimer']}"
    )


def _template_citations(packet: dict) -> list[dict]:
    """Deterministic citations matching what the template mentions."""
    exp = expected_directions(packet)
    cits = []
    strong, weak = _strong_weak(packet)
    for s in strong + weak:
        cits.append({"id": s["id"], "direction": exp[s["id"]]})
    for d in packet.get("model", {}).get("drivers", []):
        cits.append({"id": d["id"], "direction": d["direction"]})
    if "composite" in exp:
        cits.append({"id": "composite", "direction": exp["composite"]})
    if "model" in exp:
        cits.append({"id": "model", "direction": exp["model"]})
    seen, out = set(), []
    for c in cits:
        if c["id"] not in seen:
            seen.add(c["id"])
            out.append(c)
    return out


def _template_result(packet: dict, source: str, **extra) -> dict:
    text = _template(packet)
    # grounded by construction — but verified, never asserted by fiat
    leaks = check_grounding(text, packet)
    return {
        "packet": packet,
        "narrative": text,
        "reasoning": "",
        "citations": _template_citations(packet),
        "grounded": not leaks,
        "template_leaks": leaks,
        "source": source,
        **extra,
    }


def narrate_packet(packet: dict, llm=None, max_retries: int = 1) -> dict:
    """Narrate a pre-built packet under the cited-JSON contract.

    Returns {packet, narrative, reasoning, citations, grounded, source, attempts,
    first_pass_ok[, violations]}. ``llm`` is a callable(system, user) -> str.
    """
    if llm is None:
        return _template_result(packet, "template", attempts=0, first_pass_ok=True)

    user = json.dumps(packet, indent=2, default=str)
    violation_log: list[list] = []  # per-attempt violations (empty list = clean pass)
    for attempt in range(max_retries + 1):
        prompt = user if not violation_log else (
            user + "\n\nYour previous reply was REJECTED by the validator for: "
            + json.dumps([str(v) for v in violation_log[-1]])
            + ". Produce a corrected JSON reply that fixes exactly these issues."
        )
        try:
            text = llm(SYSTEM, prompt)
        except Exception as exc:  # timeout/transport: degrade, never crash the serve path
            violation_log.append([f"llm-error:{type(exc).__name__}"])
            continue
        parsed = parse_llm_json(text)
        violations = validate_narration(parsed, packet)
        violation_log.append(violations)
        if not violations:
            return {
                "packet": packet,
                "narrative": parsed["summary"].strip(),
                "reasoning": str(parsed.get("reasoning", "")),
                "citations": parsed.get("citations", []),
                "grounded": True,
                "source": "llm",
                "attempts": attempt + 1,
                "first_pass_ok": attempt == 0,
                "violation_log": violation_log,
            }
    return _template_result(
        packet, "template-fallback",
        attempts=max_retries + 1, first_pass_ok=False,
        violations=violation_log[-1], violation_log=violation_log,
    )


def narrate(company, llm=None, features_df=None, max_retries: int = 1) -> dict:
    """Build the packet for ``company`` and narrate it. ``llm`` is a callable(system, user)->str."""
    packet = build_packet(company, features_df=features_df)
    return narrate_packet(packet, llm=llm, max_retries=max_retries)
