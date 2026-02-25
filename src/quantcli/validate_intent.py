from quantcli.refusals import make_refusal
from quantcli.schemas.intent import Intent
from quantcli.schemas.refusal import Refusal
from quantcli.schemas.tool_name import ToolName


def validate_intent(intent: Intent) -> Intent | Refusal:
    """
    Validation rules (strict MVP):

    A. Exactly one ticker must be provided (single-asset only).
    B. Time range must include at least 2 trading days.
    C. Realized volatility requires a window parameter.
    D. For realized volatility, window must be strictly less than n_days.
    E. Sharpe ratio requires a window parameter.
    F. For Sharpe ratio, window must be strictly less than n_days.
    G. Window is not allowed for non-volatility metrics.
    H. risk_free_rate is only allowed for Sharpe ratio.

    Returns:
        - Intent if valid and executable
        - Refusal if the request is semantically invalid
    """
    # A. Single-asset only
    if len(intent.tickers) != 1:
        return make_refusal(
            reason="Only single-asset metrics currently supported.",
            clarifying_question="Provide exactly one ticker symbol.",
        )

    # B. Range must support returns
    if intent.time_range.n_days < 2:
        return make_refusal(
            reason="Time range must include at least 2 trading days.",
            clarifying_question="Provide time range with at least 2 trading days.",
        )

    # C + D. Realized volatility rules
    if intent.tool == ToolName.realized_volatility:
        # window parameter is required for realized volatility
        if intent.params.window is None:
            return make_refusal(
                reason="Realized volatility requires a window parameter.",
                clarifying_question="Provide window parameter for realized volatility.",
            )
        # window parameter must be less than the number of trading days in time range
        if intent.params.window >= intent.time_range.n_days:
            return make_refusal(
                reason=(
                    "Window parameter must be less than the number of trading days "
                    "in the time range."
                ),
                clarifying_question=(
                    f"Provide a window parameter less than {intent.time_range.n_days}."
                ),
            )

    # E + F. Sharpe ratio rules
    if intent.tool == ToolName.sharpe_ratio:
        # window parameter is required for Sharpe ratio
        if intent.params.window is None:
            return make_refusal(
                reason="Sharpe ratio requires a window parameter.",
                clarifying_question="Provide window parameter for Sharpe ratio.",
            )
        # window parameter must be less than the number of trading days in time range
        if intent.params.window >= intent.time_range.n_days:
            return make_refusal(
                reason=(
                    "Window parameter must be less than the number of trading days "
                    "in the time range."
                ),
                clarifying_question=(
                    f"Provide a window parameter less than {intent.time_range.n_days}."
                ),
            )

    # G. Window not allowed for other metrics
    if (
        intent.tool not in [ToolName.realized_volatility, ToolName.sharpe_ratio]
        and intent.params.window is not None
    ):
        tool_label = _tool_label(intent.tool)

        return make_refusal(
            reason=f"Window parameter is not applicable for {tool_label}.",
            clarifying_question=f"Remove window parameter for {tool_label}.",
        )

    # H. risk_free_rate only allowed for Sharpe ratio
    if intent.tool != ToolName.sharpe_ratio and intent.params.risk_free_rate != 0.0:
        tool_label = _tool_label(intent.tool)
        return make_refusal(
            reason=f"risk_free_rate parameter is not applicable for {tool_label}.",
            clarifying_question=f"Remove risk_free_rate parameter for {tool_label}.",
        )

    return intent


def _tool_label(tool: ToolName) -> str:
    return tool.value
