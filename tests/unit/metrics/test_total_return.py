import pytest
import numpy as np
from quantcli.tools.metrics import total_return
from quantcli.schemas.params import Params


def test_total_return_basic():
    prices = np.array([100, 110, 120], dtype=float)
    params = Params(window=None)
    assert total_return(prices, params) == pytest.approx(0.2, abs=1e-9)


def test_total_return_no_change():
    prices = np.array([100, 100, 100], dtype=float)
    params = Params(window=None)
    assert total_return(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_total_return_single_value():
    prices = np.array([100], dtype=float)
    params = Params(window=None)
    assert total_return(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_total_return_with_nan_middle():
    prices = np.array([100, np.nan, 120], dtype=float)
    params = Params(window=None)
    assert total_return(prices, params) == pytest.approx(0.2, abs=1e-9)


def test_total_return_with_nan_edges():
    prices = np.array([np.nan, 100, 120, np.nan], dtype=float)
    params = Params(window=None)
    assert total_return(prices, params) == pytest.approx(0.2, abs=1e-9)


def test_total_return_negative_or_zero():
    prices = np.array([100, 0, 120], dtype=float)
    params = Params(window=None)

    with pytest.raises(ValueError):
        total_return(prices, params)
    prices = np.array([100, -10, 120], dtype=float)
    with pytest.raises(ValueError):
        total_return(prices, params)


def test_total_return_all_nans():
    prices = np.array([np.nan, np.nan], dtype=float)
    params = Params(window=None)
    assert total_return(prices, params) == pytest.approx(0.0, abs=1e-9)


def test_total_return_type_and_shape_contract():
    params = Params(window=None)
    # Not a numpy array
    with pytest.raises(TypeError):
        total_return([100, 110, 120], params)
    # Not 1-D
    with pytest.raises(ValueError):
        total_return(np.array([[100, 110], [120, 130]], dtype=float), params)
