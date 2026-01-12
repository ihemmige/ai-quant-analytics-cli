import pandas as pd

def total_return(prices: pd.Series) -> float:
    """Calculate the total return of a series of prices.

    Args:
        prices (pd.Series): A series of prices.
    
    Returns:
        float: The total return of the series of prices.
    """
    if len(prices) < 2:
        return 0.0
    return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
