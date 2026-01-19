from pydantic import BaseModel


class LLMRefusal(BaseModel):
    reason: str
