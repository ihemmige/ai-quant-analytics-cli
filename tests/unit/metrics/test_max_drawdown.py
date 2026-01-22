import numpy as np
import pytest

from quantcli.schemas.params import Params
from quantcli.tools.metrics import max_drawdown


def test_max_drawdown_basic():
    prices = np.array([100, 120, 110, 90, 95, 130, 120, 80, 100], dtype=np.float64)
    params = Params(window=None)
    expected = (130 - 80) / 130
    assert max_drawdown(prices, params) == pytest.approx(expected, abs=1e-6)


def test_max_drawdown_no_drawdown():
    prices = np.array([100, 110, 120, 130, 140], dtype=np.float64)
    params = Params(window=None)
    assert max_drawdown(prices, params) == 0.0


def test_max_drawdown_all_same():
    prices = np.array([100, 100, 100, 100], dtype=np.float64)
    params = Params(window=None)
    assert max_drawdown(prices, params) == 0.0


def test_max_drawdown_type_and_shape_contract():
    params = Params(window=None)
    # Not a numpy array
    with pytest.raises(TypeError):
        max_drawdown([100, 110, 120], params)
    # Not 1-D
    with pytest.raises(ValueError):
        max_drawdown(np.array([[100, 110], [120, 130]], dtype=np.float64), params)


def test_max_drawdown_inf_raises():
    params = Params(window=None)
    with pytest.raises(ValueError):
        max_drawdown(np.array([100.0, np.inf, 120.0]), params)
    with pytest.raises(ValueError):
        max_drawdown(np.array([100.0, -np.inf, 120.0]), params)


def test_max_drawdown_with_nans_raises():
    prices = np.array([100, np.nan, 120, 110, np.nan, 90, 95], dtype=np.float64)
    params = Params(window=None)
    with pytest.raises(ValueError):
        max_drawdown(prices, params)


def test_max_drawdown_window_not_allowed():
    prices = np.array([100.0, 110.0, 90.0], dtype=np.float64)
    with pytest.raises(ValueError):
        max_drawdown(prices, Params(window=2))


def test_max_drawdown_insufficient_data_raises():
    prices = np.array([100], dtype=np.float64)
    params = Params(window=None)
    with pytest.raises(ValueError):
        max_drawdown(prices, params)


def test_max_drawdown_zero_or_negative_prices():
    params = Params(window=None)
    with pytest.raises(ValueError):
        max_drawdown(np.array([100, 0, 120], dtype=np.float64), params)
    with pytest.raises(ValueError):
        max_drawdown(np.array([100, -10, 120], dtype=np.float64), params)
