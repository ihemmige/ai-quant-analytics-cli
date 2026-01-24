import sys

import numpy as np
import pandas as pd
import pytest

from quantcli.data.price_provider import PriceProviderError
from quantcli.data.yfinance_price_provider import YFinancePriceProvider


@pytest.fixture
def patch_yfinance_history(monkeypatch):
    """Patch yfinance.Ticker(...).history(...) to return a provided DataFrame."""

    def _patch(df: pd.DataFrame, *, noisy: bool = False) -> None:
        class FakeTicker:
            def history(self, *args, **kwargs):
                if noisy:
                    print("noise to stdout")
                    print("noise to stderr", file=sys.stderr)
                return df

        import yfinance

        monkeypatch.setattr(yfinance, "Ticker", lambda _: FakeTicker())

    return _patch


def _assert_no_output(capsys: pytest.CaptureFixture[str]) -> None:
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_yfinance_provider_returns_float64_array(patch_yfinance_history, capsys):
    df = pd.DataFrame({"Close": [100.0, 105.0, 102.0, 110.0]})
    patch_yfinance_history(df, noisy=True)

    provider = YFinancePriceProvider()
    prices = provider.get_adjusted_close("AAPL", n_days=5)

    assert isinstance(prices, np.ndarray)
    assert prices.dtype == np.float64
    assert prices.shape == (4,)
    assert np.isfinite(prices).all()
    assert (prices > 0).all()

    _assert_no_output(capsys)


def test_yfinance_provider_empty_df_raises(patch_yfinance_history, capsys):
    df = pd.DataFrame()
    patch_yfinance_history(df, noisy=True)

    provider = YFinancePriceProvider()
    with pytest.raises(PriceProviderError, match="no price data"):
        provider.get_adjusted_close("AAPL", n_days=5)

    _assert_no_output(capsys)


def test_yfinance_provider_missing_close_column_raises(patch_yfinance_history, capsys):
    df = pd.DataFrame({"Open": [100.0, 101.0, 102.0]})
    patch_yfinance_history(df, noisy=True)

    provider = YFinancePriceProvider()
    with pytest.raises(PriceProviderError, match="missing close column"):
        provider.get_adjusted_close("AAPL", n_days=5)

    _assert_no_output(capsys)


def test_yfinance_provider_insufficient_points_lt_2_raises(patch_yfinance_history, capsys):
    df = pd.DataFrame({"Close": [100.0]})
    patch_yfinance_history(df, noisy=True)

    provider = YFinancePriceProvider()
    with pytest.raises(PriceProviderError, match="insufficient price points"):
        provider.get_adjusted_close("AAPL", n_days=1)

    _assert_no_output(capsys)


def test_yfinance_provider_nan_dropped_then_insufficient_points(patch_yfinance_history, capsys):
    # Provider coerces to numeric and drops NaNs; this becomes 1 point -> insufficient.
    df = pd.DataFrame({"Close": [100.0, np.nan]})
    patch_yfinance_history(df, noisy=True)

    provider = YFinancePriceProvider()
    with pytest.raises(PriceProviderError, match="insufficient price points"):
        provider.get_adjusted_close("AAPL", n_days=5)

    _assert_no_output(capsys)


def test_yfinance_provider_non_positive_prices_raise(patch_yfinance_history, capsys):
    df = pd.DataFrame({"Close": [100.0, 0.0, 105.0]})
    patch_yfinance_history(df, noisy=True)

    provider = YFinancePriceProvider()
    with pytest.raises(PriceProviderError, match="non positive prices"):
        provider.get_adjusted_close("AAPL", n_days=5)

    _assert_no_output(capsys)


def test_yfinance_provider_non_finite_prices_raise(patch_yfinance_history, capsys):
    df = pd.DataFrame({"Close": [100.0, np.inf, 105.0]})
    patch_yfinance_history(df, noisy=True)

    provider = YFinancePriceProvider()
    with pytest.raises(PriceProviderError, match="non finite prices"):
        provider.get_adjusted_close("AAPL", n_days=5)

    _assert_no_output(capsys)
