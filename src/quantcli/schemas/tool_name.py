from enum import StrEnum


class ToolName(StrEnum):
    max_drawdown = "max_drawdown"
    realized_volatility = "realized_volatility"
    total_return = "total_return"
    sharpe_ratio = "sharpe_ratio"
