import pandas as pd
import numpy as np


def max_drawdown(prices: pd.Series) -> float:
    """Compute the maximum peak-to-trough drawdown of a price series.

    Args:
        prices (pd.Series): A series of prices.

    Returns:
        float: The maximum drawdown of the series of prices, a value in [0, 1], where 0.2 corresponds to a 20% drawdown.
    """
    prices = prices.dropna()

    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive to compute drawdown.")

    if len(prices) < 2:
        return 0.0

    cumulative_max = prices.cummax()
    drawdown = (cumulative_max - prices) / cumulative_max
    return float(drawdown.max())


def total_return(prices: pd.Series) -> float:
    """Calculate the total return of a series of prices.

    Args:
        prices (pd.Series): A series of prices.

    Returns:
        float: The total return of the series of prices, where 0.1 corresponds to a 10% total return.
    """
    prices = prices.dropna()

    if len(prices) < 2:
        return 0.0

    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive to compute total return.")

    return float((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0])


def realized_volatility(
    prices: pd.Series,
    window: int,
    annualization_factor: int = 252,
) -> float:
    """Calculate the annualized realized volatility of a series of prices.

    Args:
        prices (pd.Series): A series of prices.
        window (int): window size.
        annualization_factor (int, optional): The factor to annualize the volatility. Defaults to 252.

    Returns:
        float: The annualized realized volatility of the series of prices, computed as the standard
        deviation of log returns over the specified window, annualized by the square root of the annualization factor.
    """
    prices = prices.dropna()

    if window <= 0:
        raise ValueError("Window must be a positive integer.")

    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive to compute log returns.")

    # Need at least window+1 prices to compute window log returns
    if len(prices) - 1 < window:
        return 0.0

    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(log_returns)
    window_returns = log_returns.iloc[-window:]
    print(window_returns)
    vol = window_returns.std(
        ddof=1
    )  # sample std to avoid downward bias on small windows

    print(vol)
    return float(vol * np.sqrt(annualization_factor))
