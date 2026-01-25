from quantcli.schemas.refusal import Refusal
from quantcli.tools.registry import supported_tools


def make_refusal(reason: str, clarifying_question: str | None = None) -> Refusal:
    return Refusal(
        reason=reason,
        clarifying_question=clarifying_question,
        allowed_capabilities=supported_tools(),
    )


LLM_OUTPUT_PARSE_ERROR_MESSAGE = (
    "Model output could not be parsed. "
    "Please try running the command again or rephrasing your request."
)


INTERNAL_CODE_TO_USER_REASON: dict[str, str] = {
    "LLM_OUTPUT_EMPTY": LLM_OUTPUT_PARSE_ERROR_MESSAGE,
    "LLM_OUTPUT_NOT_JSON": LLM_OUTPUT_PARSE_ERROR_MESSAGE,
    "LLM_OUTPUT_NOT_OBJECT": LLM_OUTPUT_PARSE_ERROR_MESSAGE,
    "LLM_WRAPPER_INVALID_TYPE": LLM_OUTPUT_PARSE_ERROR_MESSAGE,
    "LLM_WRAPPER_INVALID_KEYS": LLM_OUTPUT_PARSE_ERROR_MESSAGE,
    "LLM_INTENT_NOT_OBJECT": LLM_OUTPUT_PARSE_ERROR_MESSAGE,
    "LLM_INTENT_SCHEMA_INVALID": LLM_OUTPUT_PARSE_ERROR_MESSAGE,
    "LLM_REFUSAL_NOT_OBJECT": LLM_OUTPUT_PARSE_ERROR_MESSAGE,
    "LLM_REFUSAL_SCHEMA_INVALID": LLM_OUTPUT_PARSE_ERROR_MESSAGE,
    "USER_QUERY_EMPTY": "User query provided was empty. Please provide a valid query.",
}
