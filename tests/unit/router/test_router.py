from quantcli.llm.fake_llm_client import FakeLLMClient
from quantcli.router.router import route_query
from quantcli.schemas.intent import Intent
from quantcli.schemas.refusal import Refusal
from quantcli.schemas.tool_name import ToolName
from quantcli.tools.registry import supported_tools


def test_route_query_empty_input():
    user_text = "   "
    llm = FakeLLMClient("")
    refusal = route_query(user_text, llm)

    assert isinstance(refusal, Refusal)
    assert refusal.reason == "USER_QUERY_EMPTY"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None
    assert len(llm.calls) == 0


def test_route_query_valid_intent():
    user_text = "What is the max drawdown for AAPL over the last 30 days?"
    llm_response = """
    {
        "type": "intent",
        "intent": {
            "tickers": ["AAPL"],
            "time_range": {"n_days": 30},
            "tool": "max_drawdown"
        }
    }
    """
    llm = FakeLLMClient(llm_response)
    intent = route_query(user_text, llm)

    assert isinstance(intent, Intent)
    assert intent.tickers == ["AAPL"]
    assert intent.time_range.n_days == 30
    assert intent.tool == ToolName.max_drawdown
    assert intent.params.window is None
    assert intent.params.annualization_factor == 252
    assert len(llm.calls) == 1


def test_route_query_llm_refusal():
    user_text = "What is the max drawdown for AAPL over some days?"
    llm_response = """
    {
        "type": "refusal",
        "refusal": {
            "reason": "AMBIGUOUS"
        }
    }
    """
    llm = FakeLLMClient(llm_response)
    refusal = route_query(user_text, llm)

    assert isinstance(refusal, Refusal)
    assert refusal.reason == "AMBIGUOUS"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None
    assert len(llm.calls) == 1


def test_route_query_malformed_llm_response():
    user_text = (
        "What is the realized volatility for AAPL over the last 30 days "
        "with a 20-day window?"
    )
    llm_response = "not valid json"
    llm = FakeLLMClient(llm_response)
    refusal = route_query(user_text, llm)

    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_OUTPUT_NOT_JSON"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None
    assert len(llm.calls) == 1


def test_route_query_llm_exception():
    user_text = "What is the total return for AAPL over the last 30 days?"
    llm = FakeLLMClient(response=RuntimeError("LLM client error"))
    refusal = route_query(user_text, llm)

    assert isinstance(refusal, Refusal)
    assert refusal.reason == "LLM_CLIENT_ERROR"
    assert refusal.allowed_capabilities == supported_tools()
    assert refusal.clarifying_question is None
    assert len(llm.calls) == 1


def test_route_query_calls_llm_with_correct_prompts():
    user_text = "What is the total return for AAPL over the last 30 days?"
    llm_response = """
    {
        "type": "intent",
        "intent": {
            "tickers": ["AAPL"],
            "time_range": {"n_days": 30},
            "tool": "total_return"
        }
    }
    """
    llm = FakeLLMClient(llm_response)
    intent = route_query(user_text, llm)

    assert isinstance(intent, Intent)
    assert intent.tickers == ["AAPL"]
    assert intent.time_range.n_days == 30
    assert intent.tool == ToolName.total_return
    assert intent.params.window is None
    assert intent.params.annualization_factor == 252

    assert len(llm.calls) == 1
    assert llm.calls[0][0]["role"] == "system"
    assert llm.calls[0][1]["role"] == "user"
