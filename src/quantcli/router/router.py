from quantcli.llm.errors import LLMError
from quantcli.llm.llm_client import LLMClient
from quantcli.refusals import INTERNAL_CODE_TO_USER_REASON, make_refusal
from quantcli.router.decode import decode_llm_output
from quantcli.router.prompt import build_messages
from quantcli.schemas.intent import Intent
from quantcli.schemas.llm_refusal import LLMRefusal
from quantcli.schemas.refusal import Refusal


def route_query(user_text: str, llm: LLMClient) -> Intent | Refusal:
    user_text = user_text.strip()
    if user_text == "":
        return make_refusal(
            reason="USER_QUERY_EMPTY",
        )

    llm_prompts = build_messages(user_text)

    try:
        llm_output = llm.complete(llm_prompts)
    # differentiate errors for logging
    except LLMError:
        return make_refusal(
            reason="Unable to process this request right now.",
        )
    except Exception:
        return make_refusal(
            reason="Unable to process this request right now.",
        )

    decoded_output = decode_llm_output(llm_output)

    if isinstance(decoded_output, LLMRefusal):
        return make_refusal(
            reason=decoded_output.reason,
        )

    if isinstance(decoded_output, Refusal):
        mapped = INTERNAL_CODE_TO_USER_REASON.get(decoded_output.reason)
        if mapped is not None:
            return make_refusal(
                reason=mapped,
                clarifying_question=decoded_output.clarifying_question,
            )
        return decoded_output

    if isinstance(decoded_output, Intent):
        return decoded_output

    raise AssertionError("Unexpected output type from decode_llm_output")
