from quantcli.schemas import Refusal, Intent, ToolName, LLMRefusal
from quantcli.router.prompt import build_messages
from quantcli.router.decode import decode_llm_output
from quantcli.router.llm_client import LLMClient

_ALLOWED_CAPABILITIES = list(ToolName)


def route_query(user_text: str, llm: LLMClient) -> Intent | Refusal:
    user_text = user_text.strip()
    if user_text == "":
        return Refusal(
            reason="USER_QUERY_EMPTY",
            allowed_capabilities=_ALLOWED_CAPABILITIES,
            clarifying_question=None,
        )

    llm_prompts = build_messages(user_text)

    try:
        llm_output = llm.complete(llm_prompts)
    except Exception:
        return Refusal(
            reason="LLM_CLIENT_ERROR",
            allowed_capabilities=_ALLOWED_CAPABILITIES,
            clarifying_question=None,
        )

    decoded_output = decode_llm_output(llm_output)

    if isinstance(decoded_output, LLMRefusal):
        return Refusal(
            reason=decoded_output.reason,
            allowed_capabilities=_ALLOWED_CAPABILITIES,
            clarifying_question=None,
        )

    if isinstance(decoded_output, Refusal):
        return decoded_output

    if isinstance(decoded_output, Intent):
        return decoded_output

    assert False, "Unexpected output type from decode_llm_output"
