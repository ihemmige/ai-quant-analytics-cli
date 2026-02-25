import numpy as np
import pytest

from quantcli.schemas.params import Params
from quantcli.tools.metrics import sharpe_ratio


def test_sharpe_ratio_uses_last_window_returns_and_annualizes():
    # log returns are exactly [ln2, ln1.5, ln1.1]
    prices = np.array([100.0, 200.0, 300.0, 330.0], dtype=np.float64)
    window = 3
    ann = 252
    params = Params(window=window, annualization_factor=ann)

    log_returns = np.log(prices[1:] / prices[:-1])
    mean_return = float(log_returns[-window:].mean())
    vol = float(log_returns[-window:].std(ddof=1))
    expected = float((mean_return / vol) * np.sqrt(ann))

    assert sharpe_ratio(prices, params) == pytest.approx(expected, abs=1e-9)


def test_sharpe_ratio_constant_prices_raises():
    prices = np.array([100, 100, 100, 100, 100], dtype=np.float64)
    params = Params(window=3)
    with pytest.raises(ValueError):
        sharpe_ratio(prices, params)


def test_realized_volatility_constant_growth_is_zero():
    # log returns are constant => std = 0
    prices = np.array([100, 110, 121, 133.1], dtype=np.float64)
    params = Params(window=3)
    with pytest.raises(ValueError):
        sharpe_ratio(prices, params)


def test_sharpe_ratio_type_and_shape_contract():
    params = Params(window=2)
    # Not a numpy array
    with pytest.raises(TypeError):
        sharpe_ratio([100, 110, 120], params)
    # Not 1-D
    with pytest.raises(ValueError):
        sharpe_ratio(np.array([[100, 110], [120, 130]], dtype=np.float64), params)


def test_sharpe_ratio_with_nans_raises():
    params = Params(window=2)
    with pytest.raises(ValueError):
        sharpe_ratio(np.array([100.0, np.nan, 110.0], dtype=np.float64), params)


def test_sharpe_ratio_inf_raises():
    params = Params(window=2)
    with pytest.raises(ValueError):
        sharpe_ratio(np.array([100.0, np.inf, 110.0], dtype=np.float64), params)
    with pytest.raises(ValueError):
        sharpe_ratio(np.array([100.0, -np.inf, 110.0], dtype=np.float64), params)


def test_sharpe_ratio_window_required():
    prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    with pytest.raises(ValueError):
        sharpe_ratio(prices, Params(window=None))


def test_sharpe_ratio_invalid_window_raises():
    prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    with pytest.raises(ValueError):
        sharpe_ratio(prices, Params(window=1))
    with pytest.raises(ValueError):
        sharpe_ratio(prices, Params(window=0))
    with pytest.raises(ValueError):
        sharpe_ratio(prices, Params(window=-3))


def test_sharpe_ratio_raises_on_non_positive_prices():
    params = Params(window=2)
    with pytest.raises(ValueError):
        sharpe_ratio(np.array([100, 0, 110], dtype=np.float64), params)
    with pytest.raises(ValueError):
        sharpe_ratio(np.array([100, -1, 110], dtype=np.float64), params)


def test_sharpe_ratio_insufficient_data_raises():
    prices = np.array([100, 101, 102], dtype=np.float64)  # 2 returns
    params = Params(window=5)
    with pytest.raises(ValueError):
        sharpe_ratio(prices, params)


def test_sharpe_ratio_invalid_annualization_factor_raises():
    prices = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float64)
    with pytest.raises(ValueError):
        sharpe_ratio(prices, Params(window=2, annualization_factor=0))
    with pytest.raises(ValueError):
        sharpe_ratio(prices, Params(window=2, annualization_factor=-1))
    with pytest.raises(ValueError):
        sharpe_ratio(prices, Params(window=2, annualization_factor=np.inf))
    with pytest.raises(ValueError):
        sharpe_ratio(prices, Params(window=2, annualization_factor=np.nan))


def test_sharpe_ratio_with_nonzero_risk_free_rate():
    prices = np.array([100.0, 200.0, 300.0, 330.0], dtype=np.float64)
    window = 3
    ann = 252
    rf = 0.05
    params = Params(window=window, annualization_factor=ann, risk_free_rate=rf)

    log_returns = np.log(prices[1:] / prices[:-1])
    rf_daily = rf / ann
    excess = log_returns[-window:] - rf_daily
    mean_excess = float(excess.mean())
    vol = float(excess.std(ddof=1))
    expected = float((mean_excess / vol) * np.sqrt(ann))

    assert sharpe_ratio(prices, params) == pytest.approx(expected, abs=1e-9)


def test_sharpe_ratio_zero_risk_free_rate_matches_default():
    prices = np.array([100.0, 200.0, 300.0, 330.0], dtype=np.float64)
    window = 3
    ann = 252
    params_default = Params(window=window, annualization_factor=ann)
    params_explicit = Params(
        window=window, annualization_factor=ann, risk_free_rate=0.0
    )

    assert sharpe_ratio(prices, params_default) == pytest.approx(
        sharpe_ratio(prices, params_explicit), abs=1e-15
    )


def test_sharpe_ratio_inf_risk_free_rate_raises():
    prices = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float64)
    with pytest.raises(ValueError):
        sharpe_ratio(prices, Params(window=2, risk_free_rate=np.inf))
    with pytest.raises(ValueError):
        sharpe_ratio(prices, Params(window=2, risk_free_rate=np.nan))
