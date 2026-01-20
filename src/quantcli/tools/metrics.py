from __future__ import annotations

import numpy as np

from quantcli.schemas import Params  # adjust import to your actual Params location


def _clean_prices(prices: np.ndarray) -> np.ndarray:
    """
    Enforce the numeric contract for metric kernels:
    - 1-D numpy array
    - float64 dtype
    - finite values only
    """
    if not isinstance(prices, np.ndarray):
        raise TypeError("prices must be a numpy ndarray.")
    if prices.ndim != 1:
        raise ValueError("prices must be a 1-D array.")

    if prices.dtype != np.float64:
        prices = prices.astype(np.float64, copy=False)

    if not np.isfinite(prices).all():
        raise ValueError("prices must contain only finite values (no NaN/inf).")

    return prices


def total_return(prices: np.ndarray, params: Params) -> float:
    """
    Total return of a price series, where 0.1 corresponds to a 10% total return.
    """
    cleaned_prices = _clean_prices(prices)

    if params.window is not None:
        raise ValueError("Window is not supported for total_return.")

    if cleaned_prices.size < 2:
        raise ValueError(
            "At least two price points are required to compute total return."
        )

    if np.any(cleaned_prices <= 0):
        raise ValueError("Prices must be strictly positive to compute total return.")

    return float((cleaned_prices[-1] - cleaned_prices[0]) / cleaned_prices[0])


def max_drawdown(prices: np.ndarray, params: Params) -> float:
    """
    Maximum peak-to-trough drawdown of a price series, in [0, 1].
    """
    cleaned_prices = _clean_prices(prices)

    if params.window is not None:
        raise ValueError("Window is not supported for max_drawdown.")

    if cleaned_prices.size < 2:
        raise ValueError(
            "At least two price points are required to compute max drawdown."
        )

    if np.any(cleaned_prices <= 0):
        raise ValueError("Prices must be strictly positive to compute drawdown.")

    cumulative_max = np.maximum.accumulate(cleaned_prices)
    drawdown = (cumulative_max - cleaned_prices) / cumulative_max
    return float(np.max(drawdown))


def realized_volatility(prices: np.ndarray, params: Params) -> float:
    """
    Annualized realized volatility computed as the sample std (ddof=1) of log returns
    over the specified window, scaled by sqrt(annualization_factor).
    """
    cleaned_prices = _clean_prices(prices)

    window = params.window
    if window is None:
        raise ValueError("Window must be provided for realized volatility.")
    if window <= 0:
        raise ValueError("Window must be a positive integer.")
    if window < 2:
        raise ValueError("Window must be at least 2 to compute sample std (ddof=1).")

    if np.any(cleaned_prices <= 0):
        raise ValueError("Prices must be strictly positive to compute log returns.")

    # Need at least window+1 prices to compute window log returns
    if cleaned_prices.size < window + 1:
        raise ValueError(
            f"At least {window + 1} price points are required to compute realized volatility with window={window}."
        )

    log_returns = np.log(
        cleaned_prices[1:] / cleaned_prices[:-1]
    )  # length = cleaned_prices.size - 1
    window_returns = log_returns[-window:]

    vol = float(np.std(window_returns, ddof=1))
    if not np.isfinite(vol):
        raise ValueError("Computed volatility is not finite.")
    return vol * float(np.sqrt(params.annualization_factor))
