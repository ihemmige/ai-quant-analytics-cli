import numpy as np
import pandas as pd
import pytest

from quantcli.data.price_provider import PriceProviderError
from quantcli.data.yfinance_price_provider import YFinancePriceProvider


@pytest.fixture
def patch_yfinance_history(monkeypatch):
    """Patch yfinance.Ticker(...).history(...) to return a provided DataFrame (no network)."""

    def _patch(df: pd.DataFrame) -> None:
        class FakeTicker:
            def history(self, *args, **kwargs):
                return df

        import yfinance

        monkeypatch.setattr(yfinance, "Ticker", lambda _: FakeTicker())

    return _patch


def test_yfinance_provider_returns_float64_array(patch_yfinance_history):
    df = pd.DataFrame({"Close": [100.0, 105.0, 102.0, 110.0]})
    patch_yfinance_history(df)

    provider = YFinancePriceProvider()
    prices = provider.get_adjusted_close("AAPL", n_days=5)

    assert isinstance(prices, np.ndarray)
    assert prices.dtype == np.float64
    assert prices.shape == (4,)
    assert np.isfinite(prices).all()
    assert (prices > 0).all()


def test_yfinance_provider_empty_df_raises(patch_yfinance_history):
    df = pd.DataFrame()
    patch_yfinance_history(df)

    provider = YFinancePriceProvider()
    with pytest.raises(PriceProviderError):
        provider.get_adjusted_close("AAPL", n_days=5)


def test_yfinance_provider_missing_close_column_raises(patch_yfinance_history):
    df = pd.DataFrame({"Open": [100.0, 101.0, 102.0]})
    patch_yfinance_history(df)

    provider = YFinancePriceProvider()
    with pytest.raises(PriceProviderError):
        provider.get_adjusted_close("AAPL", n_days=5)


def test_yfinance_provider_nan_prices_raise(patch_yfinance_history):
    df = pd.DataFrame({"Close": [100.0, np.nan]})
    patch_yfinance_history(df)

    provider = YFinancePriceProvider()
    with pytest.raises(PriceProviderError):
        provider.get_adjusted_close("AAPL", n_days=5)


def test_yfinance_provider_non_positive_prices_raise(patch_yfinance_history):
    df = pd.DataFrame({"Close": [100.0, 0.0, 105.0]})
    patch_yfinance_history(df)

    provider = YFinancePriceProvider()
    with pytest.raises(PriceProviderError):
        provider.get_adjusted_close("AAPL", n_days=5)


def test_yfinance_provider_non_finite_prices_raise(patch_yfinance_history):
    df = pd.DataFrame({"Close": [100.0, np.inf, 105.0]})
    patch_yfinance_history(df)

    provider = YFinancePriceProvider()
    with pytest.raises(PriceProviderError):
        provider.get_adjusted_close("AAPL", n_days=5)
