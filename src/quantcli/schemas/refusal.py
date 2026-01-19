from pydantic import BaseModel, Field
from quantcli.schemas.tool_name import ToolName


class Refusal(BaseModel):
    reason: str
    clarifying_question: str | None = None
    allowed_capabilities: list[ToolName]
