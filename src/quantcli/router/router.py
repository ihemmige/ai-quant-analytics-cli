from quantcli.llm.errors import LLMError
from quantcli.llm.llm_client import LLMClient
from quantcli.refusals import INTERNAL_CODE_TO_USER_REASON, make_refusal
from quantcli.router.decode import decode_llm_output
from quantcli.router.prompt import build_messages
from quantcli.schemas.intent import Intent
from quantcli.schemas.llm_refusal import LLMRefusal
from quantcli.schemas.refusal import Refusal
from quantcli.observability.debug import log_event
import time


def route_query(user_text: str, llm: LLMClient, cid: str) -> Intent | Refusal:
    user_text = user_text.strip()
    if user_text == "":
        log_event("route_reject", cid, reason="USER_QUERY_EMPTY")
        return make_refusal(reason="USER_QUERY_EMPTY")

    llm_prompts = build_messages(user_text)
    log_event("llm_call_start", cid, prompt_count=len(llm_prompts))

    t0 = time.perf_counter()
    try:
        llm_output = llm.complete(llm_prompts)
    except LLMError as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        log_event("llm_fail", cid, elapsed_ms=elapsed_ms, kind=e.kind)
        return make_refusal(reason="Unable to process this request right now.")
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    log_event("llm_call_end", cid, elapsed_ms=elapsed_ms, output_len=len(llm_output))

    decoded_output = decode_llm_output(llm_output)

    if isinstance(decoded_output, LLMRefusal):
        log_event("llm_refusal", cid)
        return make_refusal(reason=decoded_output.reason)

    if isinstance(decoded_output, Refusal):
        log_event("decoder_reject", cid, code=decoded_output.reason)

        mapped = INTERNAL_CODE_TO_USER_REASON.get(decoded_output.reason)
        if mapped is not None:
            return make_refusal(
                reason=mapped,
                clarifying_question=decoded_output.clarifying_question,
            )
        return decoded_output

    if isinstance(decoded_output, Intent):
        log_event("intent_routed", cid, tool=decoded_output.tool.value)
        return decoded_output

    raise AssertionError("Unexpected output type from decode_llm_output")
