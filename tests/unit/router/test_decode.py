import pytest

from quantcli.router.decode import decode_llm_output
from quantcli.schemas.intent import Intent
from quantcli.schemas.llm_refusal import LLMRefusal
from quantcli.schemas.refusal import Refusal
from quantcli.schemas.tool_name import ToolName
from quantcli.tools.registry import supported_tools


def test_decode_valid_intent_no_params():
    raw = """
    {
        "type": "intent",
        "intent": {
            "tickers": ["AAPL"],
            "time_range": {"n_days": 30},
            "tool": "total_return"
        }
    }
    """
    intent = decode_llm_output(raw)
    assert intent is not None
    assert isinstance(intent, Intent)
    assert intent.tickers == ["AAPL"]
    assert intent.time_range.n_days == 30
    assert intent.tool == ToolName.total_return
    assert intent.params.window is None and intent.params.annualization_factor == 252


def test_decode_valid_intent_with_params():
    raw = """
    {
        "type": "intent",
        "intent": {
            "tickers": ["NVDA"],
            "time_range": {"n_days": 60},
            "tool": "realized_volatility",
            "params": {
                "window": 20,
                "annualization_factor": 252
            }
        }
    }
    """
    intent = decode_llm_output(raw)
    assert intent is not None
    assert isinstance(intent, Intent)
    assert intent.tickers == ["NVDA"]
    assert intent.time_range.n_days == 60
    assert intent.tool == ToolName.realized_volatility
    assert intent.params.window == 20 and intent.params.annualization_factor == 252


def test_decode_refusal():
    raw = """
    {
        "type": "refusal",
        "refusal": {
            "reason": "AMBIGUOUS"
        }
    }
    """
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, LLMRefusal)
    assert refusal.reason == "AMBIGUOUS"


def test_decode_empty_output():
    raw = " \n"
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_OUTPUT_EMPTY"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None


def test_decode_not_json():
    raw = "This is not JSON"
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_OUTPUT_NOT_JSON"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None


@pytest.mark.parametrize(
    "raw",
    [
        '[{"type": "intent", "intent": {"tickers": ["AAPL"], '
        '"time_range": {"n_days": 30}, "tool": "total_return"}}]',
        '"x"',
        "123",
    ],
)
def test_decode_not_object(raw):
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_OUTPUT_NOT_OBJECT"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None


def test_decode_missing_type():
    raw = """
    {
        "intent": {
            "tickers": ["AAPL"],
            "time_range": {"n_days": 30},
            "tool": "total_return"
        }
    }
    """
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_WRAPPER_INVALID_TYPE"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None


def test_decode_invalid_type():
    raw = """
    {
        "type": "unknown",
        "intent": {
            "tickers": ["AAPL"],
            "time_range": {"n_days": 30},
            "tool": "total_return"
        }
    }
    """
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_WRAPPER_INVALID_TYPE"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None


def test_decode_invalid_types_extra():
    raw = """
    {
        "type": "intent",
        "intent": {
            "tickers": ["AAPL"],
            "time_range": {"n_days": 30},
            "tool": "total_return"
        },
        "extra_key": "not allowed"
    }
    """
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_WRAPPER_INVALID_KEYS"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None


def test_intent_object_missing():
    raw = """
    {
        "type": "intent"
    }
    """
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_WRAPPER_INVALID_KEYS"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None


def test_refusal_object_missing():
    raw = """
    {
        "type": "refusal"
    }
    """
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_WRAPPER_INVALID_KEYS"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None


def test_intent_not_object():
    raw = """
    {
        "type": "intent",
        "intent": []
    }
    """
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_INTENT_NOT_OBJECT"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None


def test_refusal_not_object():
    raw = """
    {
        "type": "refusal",
        "refusal": []
    }
    """
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_REFUSAL_NOT_OBJECT"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None


def test_intent_schema_invalid():
    raw = """
    {
        "type": "intent",
        "intent": {
            "tickers": ["AAPL"],
            "time_range": {"n_days": "thirty"},
            "tool": "total_return"
        }
    }
    """
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_INTENT_SCHEMA_INVALID"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None


def test_refusal_schema_invalid():
    raw = """
    {
        "type": "refusal",
        "refusal": {
            "reason": 123
        }
    }
    """
    refusal = decode_llm_output(raw)
    assert refusal is not None
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_REFUSAL_SCHEMA_INVALID"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None
