from collections.abc import Sequence

from quantcli.llm.llm_client import LLMClient, Message


class FakeLLMClient(LLMClient):
    def __init__(self, response: str | Exception):
        self._response = response
        self.calls: list[Sequence[Message]] = []

    def complete(self, messages: Sequence[Message]) -> str:
        self.calls.append(messages)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response
