from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from config import DISCLAIMER
from agent.plans import get_plan, intent_label, tool_label
from agent.router import route_question
from agent.state import AgentState
from agent.tools import run_tool

PRE_TOOLS = frozenset({"filter_context", "get_article", "search_laws"})

RETRY_REASON_LABELS = {
    "empty_results": "未检索到法条",
    "low_top_score": "首轮检索置信度偏低",
    "low_score_gap": "检索结果区分度不足",
    "dispersed_laws": "跨多部法律且分数接近",
    "low_domain_confidence": "法律领域识别不确定",
    "aggressive_gap_truncate": "检索结果被过度截断",
    "topic_mismatch_高空抛物": "未检索到高空抛物相关法条",
    "topic_mismatch_未成年人": "未检索到未成年人监护责任相关法条",
}


def _agent_retry_payload(summary: dict) -> dict:
    reason = summary.get("retry_reason", "")
    return {
        "reason": reason,
        "strategy": summary.get("retry_strategy", ""),
        "attempts": summary.get("attempts", 2),
        "label": RETRY_REASON_LABELS.get(reason, "检索结果不够确定，已用原问题补充检索"),
    }


def _sse(event: str, data: dict | list | str | bool) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _plan_payload(route, plan) -> dict:
    return {
        "intent": route.intent,
        "plan_name": plan.name,
        "label": intent_label(route.intent),
        "source": route.source,
        "confidence": round(route.confidence, 2),
        "reason": route.reason,
        "steps": [{"tool": t, "label": tool_label(t)} for t in plan.steps],
    }


def _step_payload(tool: str, status: str, ms: float = 0, summary: dict | None = None) -> dict:
    payload: dict[str, Any] = {
        "tool": tool,
        "label": tool_label(tool),
        "status": status,
    }
    if ms:
        payload["ms"] = round(ms, 1)
    if summary:
        payload["summary"] = summary
    return payload


def _run_pre_tools(
    state: AgentState,
    plan,
    *,
    question: str,
    history: list[dict],
    trace,
) -> list[tuple[str, float, dict]]:
    """执行检索/查条等前置 Tool，返回 (tool, ms, summary) 列表。"""
    records: list[tuple[str, float, dict]] = []
    for tool in plan.steps:
        if tool not in PRE_TOOLS:
            continue
        t0 = time.perf_counter()
        summary = run_tool(
            tool,
            question=question,
            history=history,
            state=state,
            trace=trace,
            intent=plan.intent,
        )
        ms = (time.perf_counter() - t0) * 1000
        records.append((tool, ms, summary))

        if tool == "get_article" and plan.intent == "statute_lookup" and not summary.get("found"):
            t1 = time.perf_counter()
            fb = run_tool(
                "search_laws",
                question=question,
                history=history,
                state=state,
                trace=trace,
                intent=plan.intent,
            )
            records.append(("search_laws", (time.perf_counter() - t1) * 1000, fb))
    return records


def stream_agent_answer(
    question: str,
    history: list[dict],
    request_id: str,
    trace,
) -> Iterator[str]:
    state = AgentState(question=question, history=history, request_id=request_id)

    yield ": connected\n\n"
    yield _sse("start", {"status": "routing"})

    t0 = time.perf_counter()
    route = route_question(question, history)
    state.intent = route.intent
    state.is_legal = route.intent != "non_legal"
    plan = get_plan(route.intent)

    if trace:
        trace.step(
            "agent:route",
            (time.perf_counter() - t0) * 1000,
            {
                "intent": route.intent,
                "source": route.source,
                "confidence": route.confidence,
                "reason": route.reason,
                "plan": plan.name,
            },
        )

    t0 = time.perf_counter()
    filter_summary = run_tool(
        "filter_context",
        question=question,
        history=history,
        state=state,
        trace=trace,
        intent=plan.intent,
    )
    filter_ms = (time.perf_counter() - t0) * 1000

    yield _sse("agent_plan", _plan_payload(route, plan))
    yield _sse(
        "agent_step",
        _step_payload("filter_context", "done", filter_ms, filter_summary),
    )
    yield _sse(
        "meta",
        {
            "is_legal": state.is_legal,
            "context_turns": len(state.relevant_history),
            "request_id": request_id,
            "intent": route.intent,
        },
    )

    try:
        for tool in plan.steps:
            if tool not in PRE_TOOLS or tool == "filter_context":
                continue
            yield _sse("agent_step", _step_payload(tool, "running"))
            t0 = time.perf_counter()
            summary = run_tool(
                tool,
                question=question,
                history=history,
                state=state,
                trace=trace,
                intent=plan.intent,
            )
            ms = (time.perf_counter() - t0) * 1000
            if tool == "search_laws" and summary.get("retry"):
                yield _sse("agent_retry", _agent_retry_payload(summary))
            yield _sse("agent_step", _step_payload(tool, "done", ms, summary))

            if tool == "get_article" and plan.intent == "statute_lookup" and not summary.get("found"):
                yield _sse("agent_step", _step_payload("search_laws", "running"))
                t1 = time.perf_counter()
                fb = run_tool(
                    "search_laws",
                    question=question,
                    history=history,
                    state=state,
                    trace=trace,
                    intent=plan.intent,
                )
                yield _sse(
                    "agent_step",
                    _step_payload(
                        "search_laws",
                        "done",
                        (time.perf_counter() - t1) * 1000,
                        {"citation_count": len(state.citations), "fallback": True},
                    ),
                )

        if state.is_legal and state.intent != "non_legal":
            yield _sse("start", {"status": "retrieving" if not state.chunks else "generating"})

        yield _sse("agent_step", _step_payload("generate_answer", "running"))
        t0 = time.perf_counter()
        answer_parts: list[str] = []
        for token in state.token_stream():
            answer_parts.append(token)
            yield _sse("token", {"content": token})
        state.answer_text = "".join(answer_parts)
        gen_ms = (time.perf_counter() - t0) * 1000
        if trace:
            trace.step("generate", gen_ms, {"answer_chars": len(state.answer_text)})
        yield _sse(
            "agent_step",
            _step_payload("generate_answer", "done", gen_ms, {"answer_chars": len(state.answer_text)}),
        )

        pre_verify_answer = state.answer_text
        repair = None
        if state.is_legal and state.chunks and "verify_citations" in plan.steps:
            yield _sse("start", {"status": "verifying"})
            yield _sse("agent_step", _step_payload("verify_citations", "running"))
            t0 = time.perf_counter()
            summary = run_tool(
                "verify_citations",
                question=question,
                history=history,
                state=state,
                trace=trace,
                intent=plan.intent,
            )
            repair = state.repair
            yield _sse(
                "agent_step",
                _step_payload("verify_citations", "done", (time.perf_counter() - t0) * 1000, summary),
            )
            if repair and repair.answer != pre_verify_answer:
                state.answer_text = repair.answer
                yield _sse(
                    "answer_revision",
                    {"content": repair.answer, "action": repair.action},
                )

        if state.is_legal and state.citations:
            yield _sse("citations", state.citations)

        done_payload: dict = {
            "disclaimer": DISCLAIMER,
            "is_legal": state.is_legal,
            "citation_verified": state.citation_verified,
            "intent": route.intent,
            "agent_plan": plan.name,
            "retrieval_retry": state.retrieval_retry,
        }
        if state.retrieval_retry:
            done_payload["retrieval_retry_reason"] = state.retrieval_retry_reason
            done_payload["retrieval_retry_strategy"] = state.retrieval_retry_strategy
            done_payload["retrieval_attempts"] = state.retrieval_attempts
        if repair is not None:
            verify_data = repair.verify.to_trace_output()
            verify_data["action"] = repair.action
            verify_data["citation_verified"] = repair.citation_verified
            done_payload["citation_verify"] = verify_data

        yield _sse("done", done_payload)
        trace.finish(status="ok", is_legal=state.is_legal, answer_preview=state.answer_text)
    except Exception as exc:
        trace.finish(status="error", is_legal=state.is_legal, error=str(exc))
        yield _sse("error", {"message": str(exc)})


def run_agent_answer(
    question: str,
    history: list[dict] | None,
    trace=None,
) -> dict:
    from llm import ask_llm, ask_llm_general

    hist = history or []
    state = AgentState(question=question, history=hist, request_id="sync")
    route = route_question(question, hist)
    state.intent = route.intent
    state.is_legal = route.intent != "non_legal"
    plan = get_plan(route.intent)

    if trace:
        trace.step(
            "agent:route",
            0,
            {
                "intent": route.intent,
                "source": route.source,
                "confidence": route.confidence,
                "plan": plan.name,
            },
        )

    _run_pre_tools(state, plan, question=question, history=hist, trace=trace)

    if state.is_legal and state.chunks:
        state.answer_text = ask_llm(question, state.chunks, state.relevant_history or None)
    else:
        state.answer_text = ask_llm_general(question, state.relevant_history or None)

    if trace:
        trace.step("generate", 0, {"answer_chars": len(state.answer_text)})

    if state.is_legal and state.chunks and "verify_citations" in plan.steps:
        run_tool(
            "verify_citations",
            question=question,
            history=hist,
            state=state,
            trace=trace,
            intent=plan.intent,
        )

    return {
        "answer": state.answer_text,
        "citations": state.citations,
        "disclaimer": DISCLAIMER,
        "is_legal": state.is_legal,
        "citation_verified": state.citation_verified if state.is_legal else True,
        "intent": route.intent,
        "agent_plan": plan.name,
        "retrieval_retry": state.retrieval_retry,
        "retrieval_retry_reason": state.retrieval_retry_reason,
        "retrieval_retry_strategy": state.retrieval_retry_strategy,
        "retrieval_attempts": state.retrieval_attempts,
    }
