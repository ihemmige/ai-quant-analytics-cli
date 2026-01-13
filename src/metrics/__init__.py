"""
Metrics package: provides financial analytics functions such as max_drawdown, realized_vol, and total_return.
"""

from .max_drawdown import max_drawdown
from .total_return import total_return
# from .realized_vol import realized_vol  # TODO

__all__ = ["max_drawdown", "total_return"]
