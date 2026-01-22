import numpy as np
import pytest

from quantcli.schemas.params import Params
from quantcli.tools.metrics import realized_volatility


def test_realized_volatility_uses_last_window_returns_and_annualizes():
    # log returns are exactly [ln2, ln1.5, ln1.1]
    prices = np.array([100.0, 200.0, 300.0, 330.0], dtype=np.float64)
    window = 3
    ann = 252
    params = Params(window=window, annualization_factor=ann)

    log_returns = np.log(prices[1:] / prices[:-1])
    expected = float(log_returns[-window:].std(ddof=1) * np.sqrt(ann))

    assert realized_volatility(prices, params) == pytest.approx(expected, abs=1e-9)


def test_realized_volatility_constant_prices_is_zero():
    prices = np.array([100, 100, 100, 100, 100], dtype=np.float64)
    params = Params(window=3)
    assert realized_volatility(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_realized_volatility_constant_growth_is_zero():
    # log returns are constant => std = 0
    prices = np.array([100, 110, 121, 133.1], dtype=np.float64)
    params = Params(window=3)
    assert realized_volatility(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_realized_volatility_type_and_shape_contract():
    params = Params(window=2)
    # Not a numpy array
    with pytest.raises(TypeError):
        realized_volatility([100, 110, 120], params)
    # Not 1-D
    with pytest.raises(ValueError):
        realized_volatility(
            np.array([[100, 110], [120, 130]], dtype=np.float64), params
        )


def test_realized_volatility_with_nans_raises():
    params = Params(window=2)
    with pytest.raises(ValueError):
        realized_volatility(np.array([100.0, np.nan, 110.0], dtype=np.float64), params)


def test_realized_volatility_inf_raises():
    params = Params(window=2)
    with pytest.raises(ValueError):
        realized_volatility(np.array([100.0, np.inf, 110.0]), params)
    with pytest.raises(ValueError):
        realized_volatility(np.array([100.0, -np.inf, 110.0]), params)


def test_realized_volatility_window_required():
    prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    with pytest.raises(ValueError):
        realized_volatility(prices, Params(window=None))


def test_realized_volatility_invalid_window_raises():
    prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    with pytest.raises(ValueError):
        realized_volatility(prices, Params(window=1))
    with pytest.raises(ValueError):
        realized_volatility(prices, Params(window=0))
    with pytest.raises(ValueError):
        realized_volatility(prices, Params(window=-3))


def test_realized_volatility_raises_on_non_positive_prices():
    params = Params(window=2)
    with pytest.raises(ValueError):
        realized_volatility(np.array([100, 0, 110], dtype=np.float64), params)
    with pytest.raises(ValueError):
        realized_volatility(np.array([100, -1, 110], dtype=np.float64), params)


def test_realized_volatility_insufficient_data_raises():
    prices = np.array([100, 101, 102], dtype=np.float64)  # 2 returns
    params = Params(window=5)
    with pytest.raises(ValueError):
        realized_volatility(prices, params)
