import json

from quantcli.cli import cli
from quantcli.data.fake_price_provider import FakePriceProvider
from quantcli.llm.fake_llm_client import FakeLLMClient
from quantcli.schemas.refusal import Refusal
from quantcli.schemas.result import Result
from quantcli.schemas.tool_name import ToolName
from quantcli.tools.registry import supported_tools


def test_cli_result_exit_code_and_json(capsys, monkeypatch):
    monkeypatch.delenv("QUANTCLI_DEBUG", raising=False)
    monkeypatch.delenv("QUANTCLI_DEBUG_PATH", raising=False)

    def fake_run_query(user_query, llm_client, price_provider, cid):
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

    captured = capsys.readouterr()
    out = captured.out.strip()
    assert captured.err == ""

    assert code == 0
    parsed = json.loads(out)
    assert parsed["tool"] == "total_return"
    assert parsed["tickers"] == ["AAPL"]
    assert parsed["value"] == 0.12
    assert parsed["metadata"]["range_n_days"] == 10


def test_cli_refusal_exit_code_and_json(capsys, monkeypatch):
    monkeypatch.delenv("QUANTCLI_DEBUG", raising=False)
    monkeypatch.delenv("QUANTCLI_DEBUG_PATH", raising=False)

    def fake_run_query(user_query, llm_client, price_provider, cid):
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

    captured = capsys.readouterr()
    out = captured.out.strip()
    assert captured.err == ""

    assert code == 2
    parsed = json.loads(out)
    assert parsed["reason"] == "Only single-asset metrics currently supported."
    assert parsed["clarifying_question"] is None
    assert parsed["allowed_capabilities"] == supported_tools()


def test_cli_joins_query_tokens(capsys, monkeypatch):
    monkeypatch.delenv("QUANTCLI_DEBUG", raising=False)
    monkeypatch.delenv("QUANTCLI_DEBUG_PATH", raising=False)

    seen = {}

    def fake_run_query(user_query, llm_client, price_provider, cid):
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

    captured = capsys.readouterr()
    assert captured.err == ""
    _ = json.loads(captured.out.strip())

    assert seen["query"] == "What is max drawdown for AAPL over the past 10 days?"


def test_cli_debug_on_logs_to_stderr(capsys, monkeypatch):
    monkeypatch.setenv("QUANTCLI_DEBUG", "1")
    monkeypatch.delenv("QUANTCLI_DEBUG_PATH", raising=False)

    def fake_run_query(user_query, llm_client, price_provider, cid):
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

    captured = capsys.readouterr()

    # stdout: still exactly one JSON object
    out = captured.out.strip()
    parsed = json.loads(out)
    assert code == 0
    assert parsed["tool"] == "total_return"

    # stderr: JSON lines with event+cid
    err = captured.err.strip()
    print(err)
    assert err != ""
    for line in err.splitlines():
        rec = json.loads(line)
        assert "event" in rec
        assert "cid" in rec


def test_cli_debug_logs_to_file(tmp_path, capsys, monkeypatch):
    log_path = tmp_path / "quantcli_debug.log"

    monkeypatch.setenv("QUANTCLI_DEBUG", "1")
    monkeypatch.setenv("QUANTCLI_DEBUG_PATH", str(log_path))

    def fake_run_query(user_query, llm_client, price_provider, cid):
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

    captured = capsys.readouterr()

    # stdout: exactly one JSON object
    out = captured.out.strip()
    parsed = json.loads(out)
    assert code == 0
    assert parsed["tool"] == "total_return"

    # stderr: should be empty
    assert captured.err.strip() == ""

    # file: contains JSONL debug events
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) >= 2
    for line in lines:
        rec = json.loads(line)
        assert "event" in rec
        assert "cid" in rec
