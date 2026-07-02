"""案情首轮 query 与 retry 策略测试。"""

from __future__ import annotations

from agent.retrieval_quality import assess_retrieval_quality
from agent.retrieval_policy import resolve_case_primary_search_query
from query_rewrite import build_case_primary_query


def test_build_case_primary_query_with_topic() -> None:
    q = build_case_primary_query("未成年人从35楼往下扔水瓶砸伤路人怎么办")
    assert q is not None
    assert "抛掷" in q or "高空" in q or "侵权" in q


def test_build_case_primary_query_without_topic() -> None:
    assert build_case_primary_query("公司让我周末加班不给钱怎么办") is None


def test_resolve_case_primary_skips_when_rewrite_on() -> None:
    assert resolve_case_primary_search_query("未成年人高空抛物", rewrite=True) is None


def test_resolve_case_primary_when_rewrite_off() -> None:
    q = resolve_case_primary_search_query("未成年人高空抛物致人受伤", rewrite=False)
    assert q is not None


def test_gap_retry_skipped_when_topic_relevant_and_top_ok() -> None:
    from unittest.mock import patch

    chunks = [
        {
            "doc_id": "1254",
            "law_id": "civil_code",
            "article_no": "第一千二百五十四条",
            "score": 0.78,
        },
        {
            "doc_id": "1188",
            "law_id": "civil_code",
            "article_no": "第一千一百八十八条",
            "score": 0.74,
        },
    ]
    with patch("query_rewrite.topic_relevance_ok", return_value=(True, "")):
        q = assess_retrieval_quality(chunks, {"query_type": "case"}, question="未成年人高空抛物")
    assert q.sufficient is True
    assert q.reason == ""
