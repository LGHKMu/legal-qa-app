from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class AgentState:
    question: str
    history: list[dict]
    request_id: str
    relevant_history: list[dict] = field(default_factory=list)
    intent: str = "concept_qa"
    route_source: str = "rule"
    route_confidence: float = 0.0
    route_reason: str = ""
    is_legal: bool = True
    chunks: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    retrieve_meta: dict[str, Any] = field(default_factory=dict)
    retrieval_attempts: int = 1
    retrieval_retry: bool = False
    retrieval_retry_reason: str = ""
    retrieval_retry_strategy: str = ""
    answer_text: str = ""
    citation_verified: bool = True
    repair: Any = None
    tool_outputs: dict[str, dict] = field(default_factory=dict)

    def token_stream(self) -> Iterator[str]:
        from llm import stream_llm, stream_llm_general

        if self.is_legal and self.chunks:
            yield from stream_llm(
                self.question,
                self.chunks,
                self.relevant_history or None,
                statute_lookup=self.intent == "statute_lookup",
                case_consult=self.intent == "case_consult",
            )
        else:
            yield from stream_llm_general(
                self.question,
                self.relevant_history or None,
            )
