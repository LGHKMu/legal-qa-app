"""Agent 前置检索：与线上一致的路由 + 查条/检索工具链（供 orchestrator 评测共用）。"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.plans import get_plan
from agent.router import route_question
from agent.state import AgentState
from agent.tools import run_tool

PRE_TOOLS = frozenset({"filter_context", "get_article", "search_laws"})


@dataclass
class AgentPreRetrievalResult:
    """Agent 在完成 generate 之前产出的检索结果（与 /api/ask 线上一致）。"""

    intent: str
    route_source: str
    route_confidence: float
    route_reason: str
    is_legal: bool
    chunks: list[dict]
    retrieve_meta: dict = field(default_factory=dict)
    retrieval_retry: bool = False
    retrieval_retry_reason: str = ""
    retrieval_retry_strategy: str = ""
    retrieval_attempts: int = 1
    tools_run: list[str] = field(default_factory=list)


def run_agent_pre_retrieval(
    question: str,
    history: list[dict] | None = None,
    *,
    trace=None,
) -> AgentPreRetrievalResult:
    """执行 Agent 路由与前置检索，口径与 stream_agent_answer / run_agent_answer 相同。"""
    hist = history or []
    state = AgentState(question=question.strip(), history=hist, request_id="pre_retrieval")
    route = route_question(question, hist)
    state.intent = route.intent
    state.is_legal = route.intent != "non_legal"
    plan = get_plan(route.intent)
    tools_run: list[str] = []

    run_tool(
        "filter_context",
        question=question,
        history=hist,
        state=state,
        trace=trace,
        intent=plan.intent,
    )
    tools_run.append("filter_context")

    if state.is_legal:
        for tool in plan.steps:
            if tool not in PRE_TOOLS or tool == "filter_context":
                continue
            summary = run_tool(
                tool,
                question=question,
                history=hist,
                state=state,
                trace=trace,
                intent=plan.intent,
            )
            tools_run.append(tool)

            if (
                tool == "get_article"
                and plan.intent == "statute_lookup"
                and not summary.get("found")
            ):
                run_tool(
                    "search_laws",
                    question=question,
                    history=hist,
                    state=state,
                    trace=trace,
                    intent=plan.intent,
                )
                tools_run.append("search_laws")

    return AgentPreRetrievalResult(
        intent=route.intent,
        route_source=route.source,
        route_confidence=route.confidence,
        route_reason=route.reason,
        is_legal=state.is_legal,
        chunks=list(state.chunks),
        retrieve_meta=dict(state.retrieve_meta),
        retrieval_retry=state.retrieval_retry,
        retrieval_retry_reason=state.retrieval_retry_reason,
        retrieval_retry_strategy=state.retrieval_retry_strategy,
        retrieval_attempts=state.retrieval_attempts,
        tools_run=tools_run,
    )
