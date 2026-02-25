from quantcli.schemas.intent import Intent
from quantcli.schemas.params import Params
from quantcli.schemas.refusal import Refusal
from quantcli.schemas.time_range import TimeRange
from quantcli.schemas.tool_name import ToolName
from quantcli.validate_intent import validate_intent


def test_single_asset_only():
    intent = Intent(
        tickers=["AAPL", "GOOG"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.total_return,
        params=Params(),
    )
    result = validate_intent(intent)
    assert isinstance(result, Refusal)
    assert "single-asset" in result.reason


def test_time_range_too_short():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=1),
        tool=ToolName.total_return,
        params=Params(),
    )
    result = validate_intent(intent)
    assert isinstance(result, Refusal)
    assert "at least 2 trading days" in result.reason


def test_realized_vol_requires_window():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.realized_volatility,
        params=Params(),
    )
    result = validate_intent(intent)
    assert isinstance(result, Refusal)
    assert (
        "window parameter" in result.reason and "Realized volatility" in result.reason
    )


def test_realized_vol_window_too_large():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.realized_volatility,
        params=Params(window=5),
    )
    result = validate_intent(intent)
    assert isinstance(result, Refusal)
    assert (
        "Window parameter" in result.reason
        and "less than the number of trading days" in result.reason
    )


def test_window_not_allowed_for_non_vol_metrics():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.total_return,
        params=Params(window=3),
    )
    result = validate_intent(intent)
    assert isinstance(result, Refusal)
    assert (
        "Window parameter" in result.reason
        and "not applicable" in result.reason
        and "total_return" in result.reason
    )


def test_valid_total_return_intent_passes():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.total_return,
        params=Params(),
    )
    result = validate_intent(intent)
    assert isinstance(result, Intent)
    assert result.tool == ToolName.total_return
    assert result.tickers == ["AAPL"]
    assert result.time_range.n_days == 5
    assert result.params.window is None


def test_valid_realized_vol_intent_passes():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.realized_volatility,
        params=Params(window=3),
    )
    result = validate_intent(intent)
    assert isinstance(result, Intent)
    assert result.tool == ToolName.realized_volatility
    assert result.tickers == ["AAPL"]
    assert result.time_range.n_days == 5
    assert result.params.window == 3


def test_sharpe_ratio_requires_window():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.sharpe_ratio,
        params=Params(),
    )
    result = validate_intent(intent)
    assert isinstance(result, Refusal)
    assert "window parameter" in result.reason and "Sharpe ratio" in result.reason


def test_sharpe_ratio_window_too_large():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.sharpe_ratio,
        params=Params(window=5),
    )
    result = validate_intent(intent)
    assert isinstance(result, Refusal)
    assert (
        "Window parameter" in result.reason
        and "less than the number of trading days" in result.reason
    )


def test_valid_sharpe_ratio_intent_passes():
    intent = Intent(
        tickers=["AAPL"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.sharpe_ratio,
        params=Params(window=3),
    )
    result = validate_intent(intent)
    assert isinstance(result, Intent)
    assert result.tool == ToolName.sharpe_ratio
    assert result.tickers == ["AAPL"]
    assert result.time_range.n_days == 5
    assert result.params.window == 3
