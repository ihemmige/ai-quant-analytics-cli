import numpy as np
import pytest

from quantcli.schemas.params import Params
from quantcli.tools.metrics import total_return


def test_total_return_basic():
    prices = np.array([100, 110, 120], dtype=np.float64)
    params = Params(window=None)
    assert total_return(prices, params) == pytest.approx(0.2, abs=1e-9)


def test_total_return_no_change():
    prices = np.array([100, 100, 100], dtype=np.float64)
    params = Params(window=None)
    assert total_return(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_total_return_type_and_shape_contract():
    params = Params(window=None)
    # Not a numpy array
    with pytest.raises(TypeError):
        total_return([100, 110, 120], params)
    # Not 1-D
    with pytest.raises(ValueError):
        total_return(np.array([[100, 110], [120, 130]], dtype=np.float64), params)


def test_total_return_with_nans_raises():
    prices = np.array([100, np.nan, 120], dtype=np.float64)
    params = Params(window=None)
    with pytest.raises(ValueError):
        total_return(prices, params)


def test_total_return_with_inf_raises():
    params = Params(window=None)
    with pytest.raises(ValueError):
        total_return(np.array([100.0, np.inf, 120.0], dtype=np.float64), params)
    with pytest.raises(ValueError):
        total_return(np.array([100.0, -np.inf, 120.0], dtype=np.float64), params)


def test_total_return_window_not_allowed():
    prices = np.array([100.0, 110.0], dtype=np.float64)
    with pytest.raises(ValueError):
        total_return(prices, Params(window=2))


def test_total_return_insufficient_price_points():
    prices = np.array([100], dtype=np.float64)
    params = Params(window=None)
    with pytest.raises(ValueError):
        total_return(prices, params)


def test_total_return_negative_or_zero():
    prices = np.array([100, 0, 120], dtype=np.float64)
    params = Params(window=None)
    with pytest.raises(ValueError):
        total_return(prices, params)
    prices = np.array([100, -10, 120], dtype=np.float64)
    with pytest.raises(ValueError):
        total_return(prices, params)
