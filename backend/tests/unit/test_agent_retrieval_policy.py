"""Agent 检索策略单元测试。"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.retrieval_policy import resolve_agent_law_filter, search_rewrite_for_intent


def test_search_rewrite_disabled_for_statute_lookup() -> None:
    assert search_rewrite_for_intent("statute_lookup") is False
    assert search_rewrite_for_intent("case_consult") is None


@patch("agent.retrieval_policy.settings")
def test_law_filter_statute_lookup_uses_parse(mock_settings) -> None:
    mock_settings.agent_law_filter_enabled = True
    law_id = resolve_agent_law_filter("民法典第1046条是什么", "statute_lookup")
    assert law_id == "civil_code"


@patch("agent.retrieval_policy.settings")
def test_law_filter_case_consult_high_confidence(mock_settings) -> None:
    mock_settings.agent_law_filter_enabled = True
    mock_settings.agent_law_filter_min_confidence = 0.7
    law_id = resolve_agent_law_filter("未成年人高空抛物致人受伤怎么办", "case_consult")
    assert law_id == "civil_code"


@patch("agent.retrieval_policy.settings")
def test_law_filter_disabled_returns_none(mock_settings) -> None:
    mock_settings.agent_law_filter_enabled = False
    law_id = resolve_agent_law_filter("未成年人高空抛物致人受伤怎么办", "case_consult")
    assert law_id is None


@patch("agent.retrieval_policy.settings")
def test_law_filter_low_confidence_returns_none(mock_settings) -> None:
    mock_settings.agent_law_filter_enabled = True
    mock_settings.agent_law_filter_min_confidence = 0.95
    law_id = resolve_agent_law_filter("公司让我周末加班不给钱怎么办", "case_consult")
    assert law_id is None


def test_apply_rag_profile_fast_overrides() -> None:
    from config import Settings, apply_rag_profile

    s = Settings(
        rag_profile="fast",
        concat_retrieval_enabled=True,
        rewrite_union_rerank_enabled=True,
        rrf_pool_k=40,
        rerank_candidate_k=40,
        retrieve_candidate_k=30,
        bm25_candidate_k=20,
    )
    with patch("config.settings", s):
        apply_rag_profile()
    assert s.concat_retrieval_enabled is False
    assert s.rewrite_union_rerank_enabled is False
    assert s.rrf_pool_k == 20
    assert s.rerank_candidate_k == 25
    assert s.retrieve_candidate_k == 20
    assert s.bm25_candidate_k == 15
