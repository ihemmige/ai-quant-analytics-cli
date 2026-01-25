import atexit
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TextIO

DEBUG_ENV = "QUANTCLI_DEBUG"
DEBUG_PATH_ENV = "QUANTCLI_DEBUG_PATH"


def _utc_ts() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool
    path: str | None


def _read_config() -> DebugConfig:
    enabled = os.getenv(DEBUG_ENV, "").strip() == "1"
    path = os.getenv(DEBUG_PATH_ENV, "").strip() or None
    return DebugConfig(enabled=enabled, path=path)


class DebugLogger:
    """
    Internal-only JSONL logger.
    - Off by default (QUANTCLI_DEBUG=1 to enable)
    - Writes to stderr by default, or to QUANTCLI_DEBUG_PATH if provided
    - Never writes to stdout
    - Never raises (fails closed)
    """

    def __init__(self, config: DebugConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._fh: TextIO | None = None

        if not self._config.enabled:
            return

        # open file if provided otherwise default to stderr
        if self._config.path:
            try:
                # line-buffered text mode
                # Intentionally not using context manager: long-lived file handle
                self._fh = open(  # noqa: SIM115
                    self._config.path, "a", buffering=1, encoding="utf-8"
                )
                atexit.register(self.close)
            except Exception:
                # Fail closed: disable logging if file can't be opened
                self._fh = None
                self._config = DebugConfig(enabled=False, path=None)

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass

    def _sink(self) -> TextIO:
        return self._fh if self._fh is not None else sys.stderr

    def log_event(self, event: str, cid: str, **fields: Any) -> None:
        if not self._config.enabled:
            return

        record: dict[str, Any] = {"ts": _utc_ts(), "event": event, "cid": cid, **fields}

        try:
            line = json.dumps(record, separators=(",", ":"), default=str)
            with self._lock:
                self._sink().write(line + "\n")
        except Exception:
            pass


# Lazily initialized logger singleton
_LOGGER: DebugLogger | None = None
_LOGGER_CFG: DebugConfig | None = None


def init_logging_from_env() -> None:
    """
    Initialize the debug logger from current environment.
    """
    global _LOGGER, _LOGGER_CFG
    cfg = _read_config()

    # if already initialized with same config, do nothing
    if _LOGGER is not None and cfg == _LOGGER_CFG:
        return

    # config changed, close old file handle
    if _LOGGER is not None:
        _LOGGER.close()

    _LOGGER = DebugLogger(cfg)
    _LOGGER_CFG = cfg


def log_event(event: str, cid: str, **fields: Any) -> None:
    if _LOGGER is None:
        return
    _LOGGER.log_event(event, cid, **fields)
