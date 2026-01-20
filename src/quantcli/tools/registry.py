from typing import Callable, Mapping
import numpy as np

from quantcli.schemas.params import Params
from quantcli.schemas.tool_name import ToolName
from quantcli.tools.metrics import total_return, max_drawdown, realized_volatility

MetricFn = Callable[[np.ndarray, Params], float]

# Tools that have been implemented and exposed.
TOOL_REGISTRY: Mapping[ToolName, MetricFn] = {
    ToolName.total_return: total_return,
    ToolName.max_drawdown: max_drawdown,
    ToolName.realized_volatility: realized_volatility,
}


def supported_tools() -> list[ToolName]:
    return sorted(TOOL_REGISTRY.keys(), key=lambda t: t.value)


def get_metric(tool: ToolName) -> MetricFn | None:
    return TOOL_REGISTRY.get(tool)
