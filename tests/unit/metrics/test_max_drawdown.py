import pytest
import numpy as np
from quantcli.tools.metrics import max_drawdown
from quantcli.schemas.params import Params


def test_max_drawdown_basic():
    prices = np.array([100, 120, 110, 90, 95, 130, 120, 80, 100], dtype=float)
    params = Params(window=None)
    expected = (130 - 80) / 130
    assert max_drawdown(prices, params) == pytest.approx(expected, abs=1e-6)


def test_max_drawdown_no_drawdown():
    prices = np.array([100, 110, 120, 130, 140], dtype=float)
    params = Params(window=None)
    assert max_drawdown(prices, params) == 0.0


def test_max_drawdown_all_same():
    prices = np.array([100, 100, 100, 100], dtype=float)
    params = Params(window=None)
    assert max_drawdown(prices, params) == 0.0


def test_max_drawdown_with_nan():
    prices = np.array([100, np.nan, 120, 110, np.nan, 90, 95], dtype=float)
    params = Params(window=None)
    expected = (120 - 90) / 120
    assert max_drawdown(prices, params) == pytest.approx(expected, abs=1e-9)


def test_max_drawdown_short_series():
    prices = np.array([100], dtype=float)
    params = Params(window=None)
    assert max_drawdown(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_zero_or_negative():
    params = Params(window=None)
    with pytest.raises(ValueError):
        max_drawdown(np.array([100, 0, 120], dtype=float), params)
    with pytest.raises(ValueError):
        max_drawdown(np.array([100, -10, 120], dtype=float), params)


def test_max_drawdown_all_nan_or_one_value_after_nan_drop():
    prices = np.array([np.nan, np.nan], dtype=float)
    params = Params(window=None)
    assert max_drawdown(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_all_nans():
    prices = np.array([np.nan, np.nan], dtype=float)
    params = Params(window=None)
    assert max_drawdown(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_type_and_shape_contract():
    params = Params(window=None)
    # Not a numpy array
    with pytest.raises(TypeError):
        max_drawdown([100, 110, 120], params)
    # Not 1-D
    with pytest.raises(ValueError):
        max_drawdown(np.array([[100, 110], [120, 130]], dtype=float), params)
