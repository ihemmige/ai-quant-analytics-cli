from quantcli.schemas.refusal import Refusal
from quantcli.tools.registry import supported_tools


def make_refusal(reason: str, clarifying_question: str | None = None) -> Refusal:
    return Refusal(
        reason=reason,
        clarifying_question=clarifying_question,
        allowed_capabilities=supported_tools(),
    )
