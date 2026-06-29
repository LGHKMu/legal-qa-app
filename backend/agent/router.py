from __future__ import annotations

import json
import re
from dataclasses import dataclass

from classifier import _heuristic_non_legal
from config import settings
from llm import get_client
from query_rewrite import classify_query_type, is_article_lookup, rule_domain_hints

ROUTER_SYSTEM = """你是法律问答 Agent 的路由器。根据用户问题和对话历史，选择最合适的处理意图。
只输出 JSON，格式：
{"intent": "statute_lookup|concept_qa|case_consult|non_legal", "confidence": 0.0-1.0, "reason": "简短理由"}

意图说明：
- statute_lookup：明确查询某部法律的具体条号或条文内容
- concept_qa：概念梳理、权利概括、制度比较等
- case_consult：具体案情、纠纷、能否、怎么办、是否违法
- non_legal：与法律无关的闲聊或通用问题

若当前问题是追问且历史涉及法律，应结合历史判断。"""

VALID_INTENTS = frozenset({"non_legal", "statute_lookup", "concept_qa", "case_consult"})

CASE_ROUTE_HINTS: tuple[str, ...] = (
    "怎么办",
    "怎么处理",
    "能不能",
    "能否",
    "可以吗",
    "违法吗",
    "合法吗",
    "有没有赔偿",
)


@dataclass
class RouteResult:
    intent: str
    source: str
    confidence: float
    reason: str


def _rule_route(question: str, history: list[dict] | None) -> RouteResult | None:
    q = question.strip()
    hist = history or []

    if _heuristic_non_legal(q) and not hist:
        return RouteResult("non_legal", "rule", 0.95, "命中非法律启发式规则")

    if is_article_lookup(q):
        return RouteResult("statute_lookup", "rule", 0.95, "问题包含明确条号")

    if any(h in q for h in CASE_ROUTE_HINTS):
        return RouteResult("case_consult", "rule", 0.86, "案情咨询类问法")

    query_type = classify_query_type(q)
    if query_type == "case":
        return RouteResult("case_consult", "rule", 0.88, "案情/纠纷类问法")

    if query_type == "statute":
        return RouteResult("statute_lookup", "rule", 0.9, "查条类问法")

    if query_type == "concept":
        return RouteResult("concept_qa", "rule", 0.85, "概念/概括类问法")

    return None


def _needs_llm_router(question: str, history: list[dict] | None, rule: RouteResult | None) -> bool:
    if not settings.agent_router_llm_enabled:
        return False
    hist = history or []
    q = question.strip()

    if rule is None:
        return True

    if rule.confidence < 0.82:
        return True

    if len(q) < 15 and hist:
        return True

    hints = rule_domain_hints(q).split()
    if len(hints) >= 2:
        return True

    return False


def _llm_route(question: str, history: list[dict] | None) -> RouteResult:
    client = get_client()
    if history:
        lines = [f"{h['role']}: {h['content']}" for h in history[-6:]]
        user_content = (
            "【相关对话历史】\n"
            + "\n".join(lines)
            + f"\n\n【当前问题】\n{question}"
        )
    else:
        user_content = question

    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=128,
        )
        text = (response.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            intent = data.get("intent", "concept_qa")
            if intent not in VALID_INTENTS:
                intent = "concept_qa"
            confidence = float(data.get("confidence", 0.75))
            reason = str(data.get("reason", "LLM 路由"))
            return RouteResult(intent, "llm", confidence, reason)
    except Exception:
        pass

    return RouteResult("concept_qa", "llm", 0.6, "LLM 路由失败，回退概念解读")


def route_question(question: str, history: list[dict] | None = None) -> RouteResult:
    rule = _rule_route(question, history)
    if rule and not _needs_llm_router(question, history, rule):
        return rule
    if settings.agent_router_llm_enabled:
        llm = _llm_route(question, history)
        if rule and rule.confidence >= llm.confidence:
            return rule
        return llm
    if rule:
        return rule
    return RouteResult("concept_qa", "rule", 0.7, "默认概念解读")
