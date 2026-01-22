from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic import Anthropic, Omit, omit
from anthropic.types import MessageParam

from quantcli.llm.errors import LLMError
from quantcli.llm.llm_client import LLMClient, Message


@dataclass(frozen=True)
class AnthropicLLMClient(LLMClient):
    model: str = "claude-haiku-4-5"
    api_key: str | None = None
    max_tokens: int = 256
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    def complete(self, messages: Sequence[Message]) -> str:
        system_text, msgs = _split_messages(messages)
        used_prefill = (
            bool(msgs)
            and msgs[-1]["role"] == "assistant"
            and msgs[-1]["content"] == "{"
        )

        # Anthropic API: system is a top-level param; messages are user/assistant turns
        payload_messages: list[MessageParam] = [
            {"role": m["role"], "content": m["content"]} for m in msgs
        ]

        try:
            client = Anthropic(
                api_key=self.api_key,
                timeout=self.timeout_s,
                max_retries=0,  # no implicit retries
            )

            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_text,
                messages=payload_messages,
            )

            # extraction of model text content blocks
            raw = _extract_text(resp)
            return ("{" + raw) if used_prefill else raw

        except anthropic.RateLimitError as e:
            raise LLMError(kind="rate_limited", message="LLM rate limited.") from e
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise LLMError(kind="auth", message="LLM authentication failed.") from e
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise LLMError(kind="unavailable", message="LLM unavailable.") from e
        except anthropic.APIStatusError as e:
            # Other non-2xx (400, 404, 422, 500, 529, etc.)—treat as unavailable.
            raise LLMError(kind="unavailable", message="LLM unavailable.") from e
        except anthropic.APIError as e:
            # Catch-all for Anthropic SDK errors
            raise LLMError(kind="unavailable", message="LLM unavailable.") from e
        except Exception as e:
            # Hard boundary: never leak random exception types
            raise LLMError(kind="unavailable", message="LLM unavailable.") from e


def _split_messages(
    messages: Sequence[Message],
) -> tuple[str | Omit, list[MessageParam]]:
    system_parts: list[str] = []
    msgs: list[MessageParam] = []

    for m in messages:
        role = m["role"]
        if role == "system":
            system_parts.append(m["content"])
        elif role == "user" or role == "assistant":
            msgs.append({"role": role, "content": m["content"]})
        else:
            # Protocol prevents this, but keep deterministic behavior.
            raise ValueError(f"Unsupported message role: {role}")

    if not msgs:
        raise ValueError("At least one user message is required.")
    system_text = "\n\n".join(system_parts) if system_parts else omit
    return system_text, msgs


def _extract_text(resp: Any) -> str:
    blocks = resp.content
    if not isinstance(blocks, list) or not blocks:
        return ""

    out_parts: list[str] = []
    for b in blocks:
        if b.type == "text":
            out_parts.append(b.text)

    return "".join(out_parts)
