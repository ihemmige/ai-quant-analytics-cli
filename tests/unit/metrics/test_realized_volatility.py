import numpy as np
import pandas as pd
import pytest

from quantcli.tools.metrics import realized_volatility


def test_realized_volatility_constant_prices_is_zero():
    prices = pd.Series([100, 100, 100, 100, 100])
    assert realized_volatility(prices, window=3) == pytest.approx(0.0, abs=1e-9)


def test_realized_volatility_constant_growth_is_zero():
    # log returns are constant => std = 0
    prices = pd.Series([100, 110, 121, 133.1])
    assert realized_volatility(prices, window=3) == pytest.approx(0.0, abs=1e-9)


def test_realized_volatility_insufficient_data_returns_zero():
    prices = pd.Series([100, 101, 102])  # 2 returns total
    assert realized_volatility(prices, window=5) == pytest.approx(0.0, abs=1e-9)


def test_realized_volatility_raises_on_non_positive_prices():
    with pytest.raises(ValueError):
        realized_volatility(pd.Series([100, 0, 110]), window=2)
    with pytest.raises(ValueError):
        realized_volatility(pd.Series([100, -1, 110]), window=2)


def test_realized_volatility_window_must_be_positive():
    prices = pd.Series([100, 101, 102])
    with pytest.raises(ValueError):
        realized_volatility(prices, window=0)
    with pytest.raises(ValueError):
        realized_volatility(prices, window=-3)


def test_realized_volatility_uses_last_window_returns_and_annualizes():
    # log returns are exactly [ln2, ln1.5, ln1.1]
    prices = pd.Series([100.0, 200.0, 300.0, 330.0])
    window = 3
    ann = 252

    log_returns = np.log(prices / prices.shift(1)).dropna()
    expected = float(log_returns.iloc[-window:].std(ddof=1) * np.sqrt(ann))

    assert realized_volatility(
        prices, window=window, annualization_factor=ann
    ) == pytest.approx(expected, abs=1e-9)


def test_realized_volatility_drops_nans_in_prices():
    # After dropna: [100, 200, 300, 330] same as prior test
    prices = pd.Series([100.0, np.nan, 200.0, 300.0, 330.0, np.nan])
    window = 3

    # Build expected using the same cleaned series
    cleaned = prices.dropna()
    log_returns = np.log(cleaned / cleaned.shift(1)).dropna()
    expected = float(log_returns.iloc[-window:].std(ddof=1) * np.sqrt(252))

    assert realized_volatility(prices, window=window) == pytest.approx(
        expected, abs=1e-9
    )


def test_realized_volatility_nan_reduces_series_too_short():
    prices = pd.Series([100.0, np.nan])
    assert realized_volatility(prices, window=1) == pytest.approx(0.0, abs=1e-9)


def test_realized_volatility_drops_nans_at_edges():
    prices = pd.Series([np.nan, 100.0, 200.0, 300.0, 330.0, np.nan])
    window = 3
    cleaned = prices.dropna()
    log_returns = np.log(cleaned / cleaned.shift(1)).dropna()
    expected = float(log_returns.iloc[-window:].std(ddof=1) * np.sqrt(252))
    assert realized_volatility(prices, window=window) == pytest.approx(
        expected, abs=1e-9
    )


test_realized_volatility_uses_last_window_returns_and_annualizes()
