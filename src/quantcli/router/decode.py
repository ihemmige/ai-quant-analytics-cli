import json
from typing import Any

from pydantic import ValidationError

from quantcli.schemas import Intent, LLMRefusal, Refusal
from quantcli.tools.registry import supported_tools


def _router_refusal(reason: str) -> Refusal:
    return Refusal(
        reason=reason,
        clarifying_question=None,
        allowed_capabilities=supported_tools(),
    )


def decode_llm_output(raw: str) -> Intent | LLMRefusal | Refusal:
    s = raw.strip()
    if not s:
        return _router_refusal("LLM_OUTPUT_EMPTY")

    try:
        obj: Any = json.loads(s)
    except json.JSONDecodeError:
        return _router_refusal("LLM_OUTPUT_NOT_JSON")

    if not isinstance(obj, dict):
        return _router_refusal("LLM_OUTPUT_NOT_OBJECT")

    t = obj.get("type")
    if t not in ("intent", "refusal"):
        return _router_refusal("LLM_WRAPPER_INVALID_TYPE")

    # Optional strictness: no extra wrapper keys.
    expected_keys = {"type", "intent"} if t == "intent" else {"type", "refusal"}
    if set(obj.keys()) != expected_keys:
        return _router_refusal("LLM_WRAPPER_INVALID_KEYS")
    if t == "intent":
        payload = obj.get("intent")
        if not isinstance(payload, dict):
            return _router_refusal("LLM_INTENT_NOT_OBJECT")
        try:
            return Intent(**payload)
        except ValidationError:
            return _router_refusal("LLM_INTENT_SCHEMA_INVALID")

    payload = obj.get("refusal")
    if not isinstance(payload, dict):
        return _router_refusal("LLM_REFUSAL_NOT_OBJECT")
    try:
        return LLMRefusal(**payload)
    except ValidationError:
        return _router_refusal("LLM_REFUSAL_SCHEMA_INVALID")
