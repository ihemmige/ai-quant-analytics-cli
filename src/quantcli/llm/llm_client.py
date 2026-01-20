from __future__ import annotations

from typing import Protocol, TypedDict, Literal, Sequence


class Message(TypedDict):
    role: Literal["system", "user"]
    content: str


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Message]) -> str: ...
