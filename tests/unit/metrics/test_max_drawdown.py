import pytest
import pandas as pd
from quantcli.tools.metrics import max_drawdown


def test_max_drawdown_basic():
    prices = pd.Series([100, 120, 110, 90, 95, 130, 120, 80, 100])
    expected = (130 - 80) / 130
    assert max_drawdown(prices) == pytest.approx(expected, abs=1e-6)


def test_max_drawdown_no_drawdown():
    prices = pd.Series([100, 110, 120, 130, 140])
    assert max_drawdown(prices) == 0.0


def test_max_drawdown_all_same():
    prices = pd.Series([100, 100, 100, 100])
    assert max_drawdown(prices) == 0.0


def test_max_drawdown_with_nan():
    prices = pd.Series([100, None, 120, 110, None, 90, 95])
    expected = (120 - 90) / 120
    assert max_drawdown(prices) == pytest.approx(expected, abs=1e-9)


def test_max_drawdown_short_series():
    assert max_drawdown(pd.Series([100])) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_zero_or_negative():
    with pytest.raises(ValueError):
        max_drawdown(pd.Series([100, 0, 120]))
    with pytest.raises(ValueError):
        max_drawdown(pd.Series([100, -10, 120]))


def test_max_drawdown_all_nan_or_one_value_after_nan_drop():
    assert max_drawdown(pd.Series([None, None])) == pytest.approx(0.0, abs=1e-9)
    assert max_drawdown(pd.Series([100, None])) == pytest.approx(0.0, abs=1e-9)
