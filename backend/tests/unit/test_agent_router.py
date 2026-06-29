"""Agent Router 规则路由单元测试（不调用 LLM）。"""

from __future__ import annotations

import pytest

from agent.router import route_question
from agent.tools import lookup_article


@pytest.mark.parametrize(
    "question,expected",
    [
        ("今天天气怎么样", "non_legal"),
        ("民法典第1046条是什么", "statute_lookup"),
        ("公司让我周末加班不给钱怎么办", "case_consult"),
        ("公民的基本权利有哪些", "concept_qa"),
    ],
)
def test_rule_router(question: str, expected: str) -> None:
    result = route_question(question, [])
    assert result.intent == expected
    assert result.source == "rule"


def test_lookup_article_civil_code() -> None:
    chunk, summary = lookup_article("民法典第1046条是什么")
    assert summary.get("found") is True
    assert chunk is not None
    assert chunk["law_id"] == "civil_code"
    assert "1046" in chunk["article_no"] or "一千零四十六" in chunk["article_no"]
