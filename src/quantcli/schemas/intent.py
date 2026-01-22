from pydantic import BaseModel, Field

from quantcli.schemas.params import Params
from quantcli.schemas.time_range import TimeRange
from quantcli.schemas.tool_name import ToolName


class Intent(BaseModel):
    tool: ToolName
    tickers: list[str] = Field(min_length=1)
    time_range: TimeRange
    params: Params = Field(
        default_factory=Params,
        description="Extra parameters such as window, annualization, etc.",
    )
