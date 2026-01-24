from typing import Literal

LLMErrorKind = Literal["auth", "rate_limited", "unavailable"]


class LLMError(Exception):
    def __init__(self, kind: LLMErrorKind, message: str | None = None):
        super().__init__(kind)
        self.kind = kind
        self.message = message  # internal only

    def __str__(self) -> str:
        return f"LLMError(kind={self.kind})"
