from quantcli.llm.llm_client import LLMClient
from typing import Sequence, Mapping


class FakeLLMClient(LLMClient):
    def __init__(self, response: str | Exception):
        self._response = response
        self.calls: list[Sequence[Mapping[str, str]]] = []

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.calls.append(messages)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response
