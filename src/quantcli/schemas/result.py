from pydantic import BaseModel, Field

from quantcli.schemas.tool_name import ToolName


class Result(BaseModel):
    tool: ToolName
    tickers: list[str] = Field(min_length=1)
    value: float = Field(description="computed metric value")
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata about the result, such as time range, parameters used, etc.",
    )

    # TODO validation should ensure only one ticker for now
