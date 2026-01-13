import pytest
import pandas as pd
import numpy as np
from src.metrics.total_return import total_return

def test_total_return_basic():
    prices = pd.Series([100, 110, 120])
    assert total_return(prices) == pytest.approx(0.2)

def test_total_return_no_change():
    prices = pd.Series([100, 100, 100])
    assert total_return(prices) == pytest.approx(0.0)

def test_total_return_single_value():
    prices = pd.Series([100])
    assert total_return(prices) == 0.0

def test_total_return_with_nan_middle():
    prices = pd.Series([100, np.nan, 120])
    assert total_return(prices) == pytest.approx(0.2)

def test_total_return_with_nan_edges():
    prices = pd.Series([np.nan, 100, 120, np.nan])
    assert total_return(prices) == pytest.approx(0.2)

def test_total_return_negative_or_zero():
    prices = pd.Series([100, 0, 120])
    with pytest.raises(ValueError):
        total_return(prices)
    prices = pd.Series([100, -10, 120])
    with pytest.raises(ValueError):
        total_return(prices)
