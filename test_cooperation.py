"""Loop boundary tests, separate from model observations."""
from unittest.mock import patch
import horizon_cooperation as h


def model_message(content="", query=None, reason="stop"):
    message = {"role": "assistant", "content": content}
    if query is not None:
        message["tool_calls"] = [{"function": {
            "name": "search", "arguments": {"query": query}}}]
    return {"message": message, "done_reason": reason}


docs = [{"id": "a", "title": "Project Orion", "text": "Climate team"}]

with patch.object(h, "api") as mocked:
    result = h.generate([])
    assert result["parsed"]["abstain"] is True
    assert result["model_calls"] == 0
    assert not mocked.called

with patch.object(h, "api", return_value=model_message("I will search.")):
    evidence, trace = h.researcher("Read.", docs)
    assert evidence == []
    assert trace["tool_calls"] == 0

with patch.object(h, "api", return_value=model_message(query="Orion")):
    evidence, trace = h.researcher("Read.", docs)
    assert evidence == docs
    assert trace["stop"] == "repeated_call"
    assert trace["tool_calls"] == 1

with patch.object(h, "api", return_value=model_message(query="Orion")):
    evidence, trace = h.researcher("Read.", docs, max_tools=0)
    assert evidence == []
    assert trace["stop"] == "tool_budget"

with patch.object(h, "api", return_value=model_message(reason="length")):
    evidence, trace = h.researcher("Read.", docs)
    assert trace["stop"] == "output_limit"
    assert trace["tool_calls"] == 0

print("12 loop assertions passed; no real model call.")
