from quantcli.schemas import (
    Intent,
    Result,
    Refusal,
    ToolName,
    Params,
    TimeRange,
)
from quantcli.run import run
from quantcli.data import FakePriceProvider, PriceProviderError

import pytest

def test_invalid_intent_skips_provider():
    intent = Intent(
        tickers=["AAPL", "GOOG"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.total_return,
        params=Params(),
    )
    provider = FakePriceProvider()
    result = run(intent, provider)

    assert isinstance(result, Refusal)
    assert provider.calls == 0


def test_total_return_ok():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=10),
        tool=ToolName.total_return,
        params=Params(),
    )
    provider = FakePriceProvider()
    result = run(intent, provider)

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