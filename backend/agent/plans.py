from __future__ import annotations

from dataclasses import dataclass

INTENT_LABELS: dict[str, str] = {
    "non_legal": "一般问答",
    "statute_lookup": "法条查询",
    "concept_qa": "概念解读",
    "case_consult": "案情咨询",
}

TOOL_LABELS: dict[str, str] = {
    "filter_context": "过滤相关对话",
    "route": "识别问题意图",
    "get_article": "精确查询法条",
    "search_laws": "检索相关法条",
    "generate_answer": "生成法律回答",
    "verify_citations": "校验引用依据",
}


@dataclass(frozen=True)
class AgentPlan:
    name: str
    intent: str
    steps: tuple[str, ...]


PLANS: dict[str, AgentPlan] = {
    "non_legal": AgentPlan(
        name="non_legal",
        intent="non_legal",
        steps=("filter_context", "generate_answer"),
    ),
    "statute_lookup": AgentPlan(
        name="statute_lookup",
        intent="statute_lookup",
        steps=(
            "filter_context",
            "get_article",
            "generate_answer",
            "verify_citations",
        ),
    ),
    "concept_qa": AgentPlan(
        name="concept_qa",
        intent="concept_qa",
        steps=(
            "filter_context",
            "search_laws",
            "generate_answer",
            "verify_citations",
        ),
    ),
    "case_consult": AgentPlan(
        name="case_consult",
        intent="case_consult",
        steps=(
            "filter_context",
            "search_laws",
            "generate_answer",
            "verify_citations",
        ),
    ),
}


def get_plan(intent: str) -> AgentPlan:
    return PLANS.get(intent, PLANS["concept_qa"])


def tool_label(tool: str) -> str:
    return TOOL_LABELS.get(tool, tool)


def intent_label(intent: str) -> str:
    return INTENT_LABELS.get(intent, intent)
