from __future__ import annotations

from typing import Protocol, Sequence, Mapping


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str: ...
