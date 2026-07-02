"""Agent 前置检索与线上一致性测试。"""

from __future__ import annotations

from unittest.mock import patch

from agent.pre_retrieval import run_agent_pre_retrieval
from agent.router import route_question
from scripts.compare_rag import retrieval_hit


def test_route_statute_lookup_for_article_question() -> None:
    route = route_question("民法典第1046条是什么", [])
    assert route.intent == "statute_lookup"
    assert route.source == "rule"


def test_pre_retrieval_statute_lookup_hits_expected() -> None:
    item = {
        "law_id": "civil_code",
        "expected_articles": ["第一千零四十六条"],
    }
    outcome = run_agent_pre_retrieval("民法典第1046条是什么", [])
    assert outcome.intent == "statute_lookup"
    assert "get_article" in outcome.tools_run
    assert "search_laws" not in outcome.tools_run
    assert retrieval_hit(outcome.chunks, item)


def test_pre_retrieval_case_consult_uses_search_laws() -> None:
    with patch("agent.tools._search_laws_once") as mock_search:
        mock_search.return_value = ([], {"search_query": "test", "fusion_mode": "vector"})
        with patch("agent.retrieval_quality.assess_retrieval_quality") as mock_q:
            from agent.retrieval_quality import RetrievalQuality

            mock_q.return_value = RetrievalQuality(sufficient=True)
            outcome = run_agent_pre_retrieval("公司让我周末加班不给钱怎么办", [])
    assert outcome.intent == "case_consult"
    assert mock_search.called
    assert "search_laws" in outcome.tools_run
