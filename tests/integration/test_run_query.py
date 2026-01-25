import pytest

from quantcli.data.fake_price_provider import FakePriceProvider
from quantcli.llm.fake_llm_client import FakeLLMClient
from quantcli.orchestrator import run_query
from quantcli.schemas.refusal import Refusal
from quantcli.schemas.tool_name import ToolName
from quantcli.tools.registry import supported_tools


def test_run_query_refusal_short_circuits():
    user_query = "What is the max drawdown for AAPL over some days?"
    llm_response = """
    {
        "type": "refusal",
        "refusal": {
            "reason": "AMBIGUOUS"
        }
    }
    """
    llm_client = FakeLLMClient(llm_response)
    price_provider = FakePriceProvider()

    result = run_query(user_query, llm_client, price_provider)

    assert isinstance(result, Refusal)
    assert result.reason == "AMBIGUOUS"
    assert len(llm_client.calls) == 1
    assert price_provider.calls == 0
    assert result.allowed_capabilities == supported_tools()
    assert result.clarifying_question is None


def test_run_query_malformed_llm_response():
    user_query = "What is the max drawdown for AAPL over last 30 days?"
    llm_response = "not valid json"
    llm_client = FakeLLMClient(llm_response)
    price_provider = FakePriceProvider()

    result = run_query(user_query, llm_client, price_provider)

    assert isinstance(result, Refusal)
    assert (
        result.reason == "Model output could not be parsed. "
        "Please try running the command again or rephrasing your request."
    )
    assert len(llm_client.calls) == 1
    assert price_provider.calls == 0
    assert result.allowed_capabilities == supported_tools()
    assert result.clarifying_question is None


def test_run_query_happy_path_total_return():
    user_query = "What is the total return for AAPL over last 10 days?"
    llm_response = """
    {
        "type": "intent",
        "intent": {
            "tickers": ["AAPL"],
            "time_range": {"n_days": 10},
            "tool": "total_return"
        }
    }
    """
    llm_client = FakeLLMClient(llm_response)
    price_provider = FakePriceProvider()

    result = run_query(user_query, llm_client, price_provider)

    assert not isinstance(result, Refusal)
    assert result.tool == ToolName.total_return
    assert result.tickers == ["AAPL"]
    assert result.value == pytest.approx(0.09, abs=1e-9)
    assert result.metadata["range_n_days"] == 10
    assert result.metadata["window"] is None
    assert result.metadata["annualization_factor"] is None
    assert result.metadata["data_points"] == 10
    assert result.metadata["price_source"] == "FakePriceProvider"
    assert result.metadata["tool_version"] == "1.0.0"
    assert result.metadata["interpretation_notes"] is None
    assert len(llm_client.calls) == 1
    assert price_provider.calls == 1


def test_run_query_happy_path_max_drawdown():
    user_query = "What is the max drawdown for AAPL over last 10 days?"
    llm_response = """
    {
        "type": "intent",
        "intent": {
            "tickers": ["AAPL"],
            "time_range": {"n_days": 10},
            "tool": "max_drawdown"
        }
    }
    """
    llm_client = FakeLLMClient(llm_response)
    price_provider = FakePriceProvider("drawdown")

    result = run_query(user_query, llm_client, price_provider)

    assert not isinstance(result, Refusal)
    assert result.tool == ToolName.max_drawdown
    assert result.tickers == ["AAPL"]
    assert result.value == pytest.approx(0.33333333333333333, abs=1e-9)
    assert result.metadata["range_n_days"] == 10
    assert result.metadata["window"] is None
    assert result.metadata["annualization_factor"] is None
    assert result.metadata["data_points"] == 10
    assert result.metadata["price_source"] == "FakePriceProvider"
    assert result.metadata["tool_version"] == "1.0.0"
    assert result.metadata["interpretation_notes"] is None
    assert len(llm_client.calls) == 1
    assert price_provider.calls == 1


def test_run_query_happy_path_realized_volatility():
    user_query = (
        "What is the realized volatility for AAPL over last 10 days with a window of 5?"
    )
    llm_response = """
    {
        "type": "intent",
        "intent": {
            "tickers": ["AAPL"],
            "time_range": {"n_days": 10},
            "tool": "realized_volatility",
            "params": {
                "window": 5
            }
        }
    }
    """
    llm_client = FakeLLMClient(llm_response)
    price_provider = FakePriceProvider("monotonic_up")

    result = run_query(user_query, llm_client, price_provider)

    assert not isinstance(result, Refusal)
    assert result.tool == ToolName.realized_volatility
    assert result.tickers == ["AAPL"]
    assert result.value == pytest.approx(0.0022137963067, abs=1e-9)
    assert result.metadata["range_n_days"] == 10
    assert result.metadata["window"] == 5
    assert result.metadata["annualization_factor"] == 252
    assert result.metadata["data_points"] == 10
    assert result.metadata["price_source"] == "FakePriceProvider"
    assert result.metadata["tool_version"] == "1.0.0"
    assert result.metadata["interpretation_notes"] is None
    assert len(llm_client.calls) == 1
    assert price_provider.calls == 1


def test_run_query_intent_fails_validation():
    user_query = (
        "What is the max drawdown for AAPL over last 10 days with a window of 5?"
    )
    llm_response = """
    {
        "type": "intent",
        "intent": {
            "tickers": ["AAPL"],
            "time_range": {"n_days": 10},
            "tool": "max_drawdown",
            "params": {
                "window": 5
            }
        }
    }
    """
    llm_client = FakeLLMClient(llm_response)
    price_provider = FakePriceProvider()

    result = run_query(user_query, llm_client, price_provider)

    assert isinstance(result, Refusal)
    assert (
        "Window parameter is not applicable" in result.reason
        and "max_drawdown" in result.reason
    )
    assert len(llm_client.calls) == 1
    assert price_provider.calls == 0
