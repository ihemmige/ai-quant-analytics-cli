import pytest

from quantcli.data.fake_price_provider import FakePriceProvider
from quantcli.orchestrator import run_intent
from quantcli.schemas.intent import Intent
from quantcli.schemas.params import Params
from quantcli.schemas.refusal import Refusal
from quantcli.schemas.result import Result
from quantcli.schemas.time_range import TimeRange
from quantcli.schemas.tool_name import ToolName
from quantcli.tools.registry import TOOL_REGISTRY, supported_tools


def test_invalid_intent_skips_provider():
    intent = Intent(
        tickers=["AAPL", "GOOG"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.realized_volatility,
        params=Params(window=4, annualization_factor=252),
    )
    provider = FakePriceProvider()
    result = run_intent(intent, provider)

    assert isinstance(result, Refusal)
    assert provider.calls == 0
    assert result.allowed_capabilities == supported_tools()
    assert "single-asset metrics" in result.reason
    assert "Provide exactly one ticker symbol" in result.clarifying_question


def test_total_return_ok():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=10),
        tool=ToolName.total_return,
        params=Params(window=None),
    )
    provider = FakePriceProvider()
    result = run_intent(intent, provider)

    assert isinstance(result, Result)
    assert result.tool == ToolName.total_return
    assert result.tickers == ["AAPL"]
    assert result.value == pytest.approx(0.09, abs=1e-9)  # (109 - 100) / 100
    assert result.metadata["range_n_days"] == 10
    assert result.metadata["window"] is None
    assert result.metadata["annualization_factor"] is None
    assert result.metadata["data_points"] == 10
    assert result.metadata["price_source"] == "FakePriceProvider"
    assert result.metadata["tool_version"] == "1.0.0"
    assert result.metadata["interpretation_notes"] is None
    assert provider.calls == 1


def test_max_drawdown_ok():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=10),
        tool=ToolName.max_drawdown,
        params=Params(window=None),
    )
    provider = FakePriceProvider("drawdown")
    result = run_intent(intent, provider)

    assert isinstance(result, Result)
    assert result.tool == ToolName.max_drawdown
    assert result.tickers == ["AAPL"]
    assert result.value == pytest.approx(
        0.33333333333333333, abs=1e-9
    )  # (130 - 89.99999999999999) / 130
    assert result.metadata["range_n_days"] == 10
    assert result.metadata["window"] is None
    assert result.metadata["annualization_factor"] is None
    assert result.metadata["data_points"] == 10
    assert result.metadata["price_source"] == "FakePriceProvider"
    assert result.metadata["tool_version"] == "1.0.0"
    assert result.metadata["interpretation_notes"] is None
    assert provider.calls == 1


def test_realized_volatility_ok():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=10),
        tool=ToolName.realized_volatility,
        params=Params(window=5, annualization_factor=252),
    )
    provider = FakePriceProvider("monotonic_up")
    result = run_intent(intent, provider)

    assert isinstance(result, Result)
    assert result.tool == ToolName.realized_volatility
    assert result.tickers == ["AAPL"]
    assert result.value == pytest.approx(
        0.0022137963067, abs=1e-9
    )  # std of log returns * sqrt(252)
    assert result.metadata["range_n_days"] == 10
    assert result.metadata["window"] == 5
    assert result.metadata["annualization_factor"] == 252
    assert result.metadata["data_points"] == 10
    assert result.metadata["price_source"] == "FakePriceProvider"
    assert result.metadata["tool_version"] == "1.0.0"
    assert result.metadata["interpretation_notes"] is None
    assert provider.calls == 1


def test_provider_failure_returns_refusal():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=10),
        tool=ToolName.max_drawdown,
        params=Params(window=None),
    )
    provider = FakePriceProvider(fail=True)
    result = run_intent(intent, provider)

    assert isinstance(result, Refusal)
    assert result.reason == "Unable to retrieve valid price data."
    assert result.allowed_capabilities == supported_tools()
    assert provider.calls == 1


def test_result_metadata_complete():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=10),
        tool=ToolName.realized_volatility,
        params=Params(window=7, annualization_factor=252),
    )
    provider = FakePriceProvider()
    result = run_intent(intent, provider)

    assert isinstance(result, Result)
    assert "range_n_days" in result.metadata
    assert "window" in result.metadata
    assert "annualization_factor" in result.metadata
    assert "data_points" in result.metadata
    assert "price_source" in result.metadata
    assert "tool_version" in result.metadata
    assert "interpretation_notes" in result.metadata


def test_all_registry_defined_tools_wired():
    assert set(TOOL_REGISTRY.keys()).issubset(set(ToolName))

    for tool in TOOL_REGISTRY.keys():
        intent = Intent(
            tickers=["AAPL"],
            time_range=TimeRange(n_days=10),
            tool=tool,
            params=(
                Params(window=5, annualization_factor=252)
                if tool == ToolName.realized_volatility
                else Params(window=None)
            ),
        )
        provider = FakePriceProvider()
        result = run_intent(intent, provider)

        assert isinstance(result, Result)
        assert result.tool == tool
