"""Agent 检索策略：法律域硬过滤、查条短路参数。"""

from __future__ import annotations

from config import settings
from query_rewrite import infer_law_filter_from_rules


def resolve_agent_law_filter(question: str, intent: str) -> str | None:
    """为 Agent 检索解析 law_filter（硬过滤 Chroma/BM25 范围）。"""
    if not settings.agent_law_filter_enabled:
        return None

    if intent == "statute_lookup":
        from agent.tools import _parse_law_id

        return _parse_law_id(question)

    law_id, conf = infer_law_filter_from_rules(question)
    if law_id and conf >= settings.agent_law_filter_min_confidence:
        return law_id
    return None


def search_rewrite_for_intent(intent: str) -> bool | None:
    """查条 fallback 检索不走 LLM 改写，概念/案情走默认改写。"""
    if intent == "statute_lookup":
        return False
    return None


def resolve_case_primary_search_query(question: str, *, rewrite: bool | None) -> str | None:
    """案情首轮：无 LLM 改写且命中主题规则时，用规则 query 替代口语原问。"""
    if rewrite:
        return None
    from query_rewrite import build_case_primary_query

    return build_case_primary_query(question)
