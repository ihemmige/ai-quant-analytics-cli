from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Message]) -> str: ...
