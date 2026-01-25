import json

from quantcli.observability.debug import DebugConfig, DebugLogger


def test_debug_logger_disabled_writes_nothing(capsys, cid):
    logger = DebugLogger(DebugConfig(enabled=False, path=None))
    logger.log_event("test_event", cid=cid, foo=1)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_debug_logger_enabled_writes_jsonl_to_stderr(capsys, cid):
    logger = DebugLogger(DebugConfig(enabled=True, path=None))
    logger.log_event("test_event", cid=cid, foo=1)

    captured = capsys.readouterr()
    assert captured.out == ""

    err = captured.err.strip()
    assert err != ""
    lines = err.splitlines()
    assert len(lines) == 1

    rec = json.loads(lines[0])
    assert rec["event"] == "test_event"
    assert rec["cid"] == cid
    assert rec["foo"] == 1
    assert "ts" in rec


def test_debug_logger_enabled_writes_jsonl_to_file(tmp_path, cid):
    log_path = tmp_path / "debug.log"
    logger = DebugLogger(DebugConfig(enabled=True, path=str(log_path)))

    logger.log_event("test_event", cid=cid, foo=1)
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1

    rec = json.loads(lines[0])
    assert rec["event"] == "test_event"
    assert rec["cid"] == cid
    assert rec["foo"] == 1
    assert "ts" in rec
