from quantcli.schemas import (
    Intent,
    Result,
    Refusal,
    ToolName,
    Params,
    TimeRange,
)
from quantcli.orchestrator import run_intent
from quantcli.data import FakePriceProvider, PriceProviderError

import pytest


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
    assert result.allowed_capabilities == list(ToolName)
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
    assert "Failed to fetch price data" in result.reason
    assert result.allowed_capabilities == list(ToolName)
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


def test_all_tools_wired():
    for tool in ToolName:
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
