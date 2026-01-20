from dataclasses import dataclass
from typing import Literal, Optional


LLMErrorKind = Literal["auth", "rate_limited", "unavailable"]


@dataclass(frozen=True)
class LLMError(Exception):
    kind: LLMErrorKind
    message: str

    def __str__(self) -> str:
        return f"LLMError(kind={self.kind}): {self.message}"
