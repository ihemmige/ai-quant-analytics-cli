from enum import Enum

class ToolName(str, Enum):
    max_drawdown = "max_drawdown"
    realized_volatility = "realized_volatility"
    total_return = "total_return"