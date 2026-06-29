"""主题检索锚词与相关性规则测试。"""

from __future__ import annotations

from agent.retrieval_quality import assess_retrieval_quality
from query_rewrite import (
    build_case_retry_query,
    classify_query_type,
    enrich_search_query,
    topic_relevance_ok,
    topic_search_hints,
)


def test_classify_short_case_consult_as_case() -> None:
    assert classify_query_type("未成年人高空抛物致人受伤怎么办") == "case"


def test_topic_hints_for_high_parabola() -> None:
    hints = topic_search_hints("未成年人高空抛物致人受伤怎么办")
    assert "抛掷物品" in hints
    assert "监护人责任" in hints


def test_enrich_search_query_adds_disambiguation() -> None:
    q = enrich_search_query("高空抛物 未成年人 侵权责任 民法典", "未成年人高空抛物致人受伤怎么办")
    assert "抛掷物品" in q


def test_build_case_retry_query_uses_topic_hints() -> None:
    q = build_case_retry_query("未成年人高空抛物致人受伤怎么办", {})
    assert "抛掷物品" in q
    assert "怎么办" not in q


def test_topic_relevance_detects_wrong_high_altitude_articles() -> None:
    chunks = [
        {
            "article_no": "第一千二百四十条",
            "text": "从事高空、高压、地下挖掘活动造成他人损害的",
        }
    ]
    ok, reason = topic_relevance_ok("未成年人高空抛物致人受伤怎么办", chunks)
    assert ok is False
    assert reason.startswith("topic_mismatch_")


def test_assess_quality_flags_topic_mismatch() -> None:
    chunks = [
        {
            "article_no": "第一千二百四十条",
            "text": "从事高空、高压、地下挖掘活动造成他人损害的",
            "rerank_score": 0.99,
        },
        {
            "article_no": "第一千二百四十一条",
            "text": "遗失、抛弃高度危险物造成他人损害的",
            "rerank_score": 0.98,
        },
    ]
    q = assess_retrieval_quality(
        chunks,
        {"query_type": "case"},
        question="未成年人高空抛物致人受伤怎么办",
    )
    assert q.sufficient is False
    assert q.reason.startswith("topic_mismatch_")
