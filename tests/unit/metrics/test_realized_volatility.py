import numpy as np
import pytest
from quantcli.tools.metrics import realized_volatility
from quantcli.schemas import Params


def test_realized_volatility_constant_prices_is_zero():
    prices = np.array([100, 100, 100, 100, 100], dtype=float)
    params = Params(window=3)
    assert realized_volatility(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_realized_volatility_constant_growth_is_zero():
    # log returns are constant => std = 0
    prices = np.array([100, 110, 121, 133.1], dtype=float)
    params = Params(window=3)
    assert realized_volatility(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_realized_volatility_insufficient_data_returns_zero():
    prices = np.array([100, 101, 102], dtype=float)  # 2 returns total
    params = Params(window=5)
    assert realized_volatility(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_realized_volatility_raises_on_non_positive_prices():
    params = Params(window=2)
    with pytest.raises(ValueError):
        realized_volatility(np.array([100, 0, 110], dtype=float), params)
    with pytest.raises(ValueError):
        realized_volatility(np.array([100, -1, 110], dtype=float), params)


def test_realized_volatility_window_must_be_positive():
    prices = np.array([100, 101, 102], dtype=float)
    with pytest.raises(ValueError):
        realized_volatility(prices, Params(window=0))
    with pytest.raises(ValueError):
        realized_volatility(prices, Params(window=-3))


def test_realized_volatility_uses_last_window_returns_and_annualizes():
    # log returns are exactly [ln2, ln1.5, ln1.1]
    prices = np.array([100.0, 200.0, 300.0, 330.0], dtype=float)
    window = 3
    ann = 252
    params = Params(window=window, annualization_factor=ann)

    log_returns = np.log(prices[1:] / prices[:-1])
    expected = float(log_returns[-window:].std(ddof=1) * np.sqrt(ann))

    assert realized_volatility(prices, params) == pytest.approx(expected, abs=1e-9)


def test_realized_volatility_drops_nans_in_prices():
    # After dropna: [100, 200, 300, 330] same as prior test
    prices = np.array([100.0, np.nan, 200.0, 300.0, 330.0, np.nan], dtype=float)
    window = 3
    params = Params(window=window)

    # Build expected using the same cleaned series
    cleaned = prices[np.isfinite(prices)]
    log_returns = np.log(cleaned[1:] / cleaned[:-1])
    expected = float(log_returns[-window:].std(ddof=1) * np.sqrt(252))

    assert realized_volatility(prices, params) == pytest.approx(expected, abs=1e-9)


def test_realized_volatility_nan_reduces_series_too_short():
    prices = np.array([100.0, np.nan], dtype=float)
    params = Params(window=1)
    assert realized_volatility(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_realized_volatility_drops_nans_at_edges():
    prices = np.array([np.nan, 100.0, 200.0, 300.0, 330.0, np.nan], dtype=float)
    window = 3
    params = Params(window=window)
    cleaned = prices[np.isfinite(prices)]
    log_returns = np.log(cleaned[1:] / cleaned[:-1])
    expected = float(log_returns[-window:].std(ddof=1) * np.sqrt(252))
    assert realized_volatility(prices, params) == pytest.approx(expected, abs=1e-9)


def test_realized_volatility_type_and_shape_contract():
    params = Params(window=2)
    # Not a numpy array
    with pytest.raises(TypeError):
        realized_volatility([100, 110, 120], params)
    # Not 1-D
    with pytest.raises(ValueError):
        realized_volatility(np.array([[100, 110], [120, 130]], dtype=float), params)


def test_realized_volatility_all_nans():
    prices = np.array([np.nan, np.nan], dtype=float)
    params = Params(window=2)
    assert realized_volatility(prices, params) == pytest.approx(0.0, abs=1e-9)
