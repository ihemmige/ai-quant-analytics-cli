from typing import Protocol, TypedDict, Literal, Sequence


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Message]) -> str: ...
