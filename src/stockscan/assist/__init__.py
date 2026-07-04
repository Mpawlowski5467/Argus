"""AI assist — read-only, grounded LLM helpers (FIREWALLED from the signal).

Nothing here is a feature, a score, or a point-in-time input to the panel. These are
consumers of already-computed, deterministic data (the narration packet, the ops
state, a git diff) that the local model explains or reviews. The founding guarantee
carries over from narration: numbers come from CODE, the model only frames them, and
every numeral in any answer must trace to the provided context (``core.grounded_answer``
reuses the narration grounding guard).

  core   -> grounded_answer: answer STRICTLY from a context dict, or refuse
  qa     -> conversational Q&A over a ticker's packet + news memory
  judge  -> LLM faithfulness judge (the paraphrase-level check the guard can't do)
  audit  -> firewall / look-ahead auditor (deterministic import scan + LLM diff review)
  brief  -> nightly natural-language ops digest
"""

from .core import grounded_answer

__all__ = ["grounded_answer"]
