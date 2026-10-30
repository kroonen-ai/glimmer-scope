import io
import json

import pytest

import scope.llm as llm


def replies(monkeypatch, answers):
    sent = []

    def fake_urlopen(request, timeout):
        sent.append(json.loads(request.data))
        content, reason = answers.pop(0)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}, "finish_reason": reason}]}).encode())

    monkeypatch.setenv("META_API_KEY", "test")
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    return sent


def test_empty_reply_is_asked_again_with_more_room(monkeypatch):
    sent = replies(monkeypatch, [("", "length"), ("[]", "stop")])
    assert llm.chat("hi", max_tokens=4000) == "[]"
    assert [s["max_tokens"] for s in sent] == [4000, 8000]


def test_reply_that_stays_empty_names_the_reason(monkeypatch):
    replies(monkeypatch, [("", "content_filter")] * 3)
    with pytest.raises(RuntimeError, match="empty 3 times.*content_filter"):
        llm.chat("hi")
