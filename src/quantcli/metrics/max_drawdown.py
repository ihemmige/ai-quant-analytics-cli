import pandas as pd

def max_drawdown(prices: pd.Series) -> float:
    """Calculate the maximum drawdown of a series of prices.

    Args:
        prices (pd.Series): A series of prices.
    
    Returns:
        float: The maximum drawdown of the series of prices.
    """
    if len(prices) < 2:
        return 0.0
    cumulative_max = prices.cummax()
    drawdown = (cumulative_max - prices) / cumulative_max
    return drawdown.max()
