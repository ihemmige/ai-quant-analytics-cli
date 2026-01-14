import pandas as pd


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
