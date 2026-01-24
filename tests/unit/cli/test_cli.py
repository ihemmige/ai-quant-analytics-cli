import json

from quantcli.cli import cli
from quantcli.data.fake_price_provider import FakePriceProvider
from quantcli.llm.fake_llm_client import FakeLLMClient
from quantcli.schemas.refusal import Refusal
from quantcli.schemas.result import Result
from quantcli.schemas.tool_name import ToolName
from quantcli.tools.registry import supported_tools


def test_cli_result_exit_code_and_json(capsys):
    def fake_run_query(user_query, llm_client, price_provider):
        return Result(
            tool=ToolName.total_return,
            tickers=["AAPL"],
            value=0.12,
            metadata={"range_n_days": 10},
        )

    code = cli(
        ["total", "return", "AAPL", "10", "days"],
        llm_factory=lambda: FakeLLMClient("valid response"),
        provider_factory=FakePriceProvider,
        run_query_fn=fake_run_query,
    )
    out = capsys.readouterr().out.strip()

    assert code == 0
    parsed = json.loads(out)
    assert parsed["tool"] == "total_return"
    assert parsed["tickers"] == ["AAPL"]
    assert parsed["value"] == 0.12
    assert parsed["metadata"]["range_n_days"] == 10


def test_cli_refusal_exit_code_and_json(capsys):
    def fake_run_query(user_query, llm_client, price_provider):
        return Refusal(
            reason="Only single-asset metrics currently supported.",
            clarifying_question=None,
            allowed_capabilities=supported_tools(),
        )

    code = cli(
        ["compare", "AAPL", "and", "MSFT"],
        llm_factory=lambda: FakeLLMClient("valid response"),
        provider_factory=FakePriceProvider,
        run_query_fn=fake_run_query,
    )
    out = capsys.readouterr().out.strip()

    assert code == 2
    parsed = json.loads(out)
    assert parsed["reason"] == "Only single-asset metrics currently supported."
    assert parsed["clarifying_question"] is None
    assert parsed["allowed_capabilities"] == supported_tools()


def test_cli_joins_query_tokens(capsys):
    seen = {}

    def fake_run_query(user_query, llm_client, price_provider):
        seen["query"] = user_query
        return Refusal(
            reason="x",
            clarifying_question=None,
            allowed_capabilities=[],
        )

    _ = cli(
        [
            "What",
            "is",
            "max",
            "drawdown",
            "for",
            "AAPL",
            "over",
            "the",
            "past",
            "10",
            "days?",
        ],
        llm_factory=lambda: FakeLLMClient("valid response"),
        provider_factory=FakePriceProvider,
        run_query_fn=fake_run_query,
    )
    _ = capsys.readouterr()  # consume stdout; not asserted here

    assert seen["query"] == "What is max drawdown for AAPL over the past 10 days?"
