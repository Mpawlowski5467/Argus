"""LocalLLM payload contract: the chat cap goes out as max_tokens; narration stays
uncapped. Transport is stubbed at the httpx-client level — no network in tests."""

from stockscan.narrate.llm import LocalLLM
from stockscan.view.data import ArgusData


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}],
                "usage": {"completion_tokens": 2}}


class _Client:
    def __init__(self):
        self.sent = None

    def post(self, url, json=None):
        self.sent = json
        return _Resp()


def test_max_tokens_rides_in_the_payload_only_when_set():
    llm = LocalLLM(model="m", max_tokens=500)
    llm._client = _Client()
    assert llm("sys", "user") == "ok"
    assert llm._client.sent["max_tokens"] == 500

    uncapped = LocalLLM(model="m")            # the narrator's configuration
    uncapped._client = _Client()
    uncapped("sys", "user")
    assert "max_tokens" not in uncapped._client.sent


def test_reasoning_effort_rides_in_the_payload_only_when_set():
    llm = LocalLLM(model="m", reasoning_effort="none")
    llm._client = _Client()
    llm("sys", "user")
    assert llm._client.sent["reasoning_effort"] == "none"

    default = LocalLLM(model="m")
    default._client = _Client()
    default("sys", "user")
    assert "reasoning_effort" not in default._client.sent


def test_chat_llm_uses_the_chat_knobs(monkeypatch):
    import stockscan.config as config

    monkeypatch.setattr(config, "LLM_CHAT_MODEL", "tiny-model")
    monkeypatch.setattr(config, "LLM_CHAT_MAX_TOKENS", 123)
    monkeypatch.setattr(config, "LLM_CHAT_TIMEOUT", 45.0)
    monkeypatch.setattr(config, "LLM_CHAT_REASONING", "none")
    llm = ArgusData._chat_llm()
    assert llm.model == "tiny-model" and llm.max_tokens == 123
    assert llm.reasoning_effort == "none"


def test_empty_response_refuses_instead_of_blank_bubble():
    from stockscan.assist.core import REFUSAL, grounded_answer

    calls = []

    def llm(system, user):
        calls.append(user)
        return "   "        # a capped thinking model: all budget spent on reasoning

    r = grounded_answer({"meta": {"x": 1}}, "why?", llm, "sys")
    assert r["refused"] is True and r["answer"] == REFUSAL
    assert r["violations"] == ["empty-response"]
    assert len(calls) == 2   # empty is retryable (cheap), unlike a transport timeout
