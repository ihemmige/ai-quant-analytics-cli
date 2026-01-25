import numpy as np
from numpy.typing import NDArray

from quantcli.schemas.params import Params


def _validate_prices(prices: NDArray[np.float64]) -> NDArray[np.float64]:
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


def total_return(prices: NDArray[np.float64], params: Params) -> float:
    """
    Total return of a price series, where 0.1 corresponds to a 10% total return.
    """

    validated_prices = _validate_prices(prices)

    if params.window is not None:
        raise ValueError("Window is not supported for total_return.")

    if validated_prices.size < 2:
        raise ValueError(
            "At least two price points are required to compute total return."
        )

    if np.any(validated_prices <= 0):
        raise ValueError("Prices must be strictly positive to compute total return.")

    return float((validated_prices[-1] - validated_prices[0]) / validated_prices[0])


def max_drawdown(prices: NDArray[np.float64], params: Params) -> float:
    """
    Maximum peak-to-trough drawdown of a price series, in [0, 1].
    """

    validated_prices = _validate_prices(prices)

    if params.window is not None:
        raise ValueError("Window is not supported for max_drawdown.")

    if validated_prices.size < 2:
        raise ValueError(
            "At least two price points are required to compute max drawdown."
        )

    if np.any(validated_prices <= 0):
        raise ValueError("Prices must be strictly positive to compute drawdown.")

    cumulative_max = np.maximum.accumulate(validated_prices)
    drawdown = (cumulative_max - validated_prices) / cumulative_max
    return float(np.max(drawdown))


def realized_volatility(prices: NDArray[np.float64], params: Params) -> float:
    """
    Annualized realized volatility computed as the sample std (ddof=1) of log returns
    over the specified window, scaled by sqrt(annualization_factor).
    """

    validated_prices = _validate_prices(prices)

    window = params.window
    if window is None:
        raise ValueError("Window must be provided for realized volatility.")
    if window <= 0:
        raise ValueError("Window must be a positive integer.")
    if window < 2:
        raise ValueError("Window must be at least 2 to compute sample std (ddof=1).")

    if np.any(validated_prices <= 0):
        raise ValueError("Prices must be strictly positive to compute log returns.")

    # Need at least window+1 prices to compute window log returns
    if validated_prices.size < window + 1:
        raise ValueError(
            f"At least {window + 1} price points are required to compute realized "
            f"volatility with window={window}."
        )

    log_returns = np.log(
        validated_prices[1:] / validated_prices[:-1]
    )  # length = validated_prices.size - 1
    window_returns = log_returns[-window:]

    vol = float(np.std(window_returns, ddof=1))
    if not np.isfinite(vol):
        raise ValueError("Computed volatility is not finite.")
    return vol * float(np.sqrt(params.annualization_factor))
