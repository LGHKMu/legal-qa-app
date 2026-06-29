"""案情检索质量评估与 merge 单元测试。"""

from __future__ import annotations

from agent.retrieval_quality import assess_retrieval_quality, merge_retrieval_chunks


def _chunk(doc_id: str, score: float, law_id: str = "labor_law") -> dict:
    return {
        "doc_id": doc_id,
        "law_id": law_id,
        "law_name": "劳动法",
        "article_no": doc_id,
        "score": score,
    }


def test_empty_results_not_sufficient() -> None:
    q = assess_retrieval_quality([], {})
    assert q.sufficient is False
    assert q.reason == "empty_results"


def test_high_confidence_sufficient() -> None:
    chunks = [_chunk("a1", 0.82), _chunk("a2", 0.65)]
    q = assess_retrieval_quality(chunks, {"query_type": "case", "domain_confidence": 0.9})
    assert q.sufficient is True
    assert q.reason == ""


def test_low_top_score_triggers_retry() -> None:
    chunks = [_chunk("a1", 0.42), _chunk("a2", 0.40)]
    q = assess_retrieval_quality(chunks, {})
    assert q.sufficient is False
    assert q.reason == "low_top_score"


def test_low_score_gap_triggers_retry() -> None:
    chunks = [_chunk("a1", 0.72), _chunk("a2", 0.69)]
    q = assess_retrieval_quality(chunks, {})
    assert q.sufficient is False
    assert q.reason == "low_score_gap"


def test_low_domain_confidence_for_case() -> None:
    chunks = [_chunk("a1", 0.80), _chunk("a2", 0.60)]
    q = assess_retrieval_quality(
        chunks,
        {"query_type": "case", "domain_confidence": 0.55},
    )
    assert q.sufficient is False
    assert q.reason == "low_domain_confidence"


def test_merge_dedupes_and_keeps_primary_order() -> None:
    primary = [_chunk("a1", 0.9), _chunk("a2", 0.8)]
    secondary = [_chunk("a2", 0.85), _chunk("a3", 0.7)]
    merged = merge_retrieval_chunks(primary, secondary, final_k=5)
    assert [c["doc_id"] for c in merged] == ["a1", "a2", "a3"]


def test_merge_truncates_to_final_k() -> None:
    primary = [_chunk("a1", 0.9), _chunk("a2", 0.8), _chunk("a3", 0.7)]
    secondary = [_chunk("a4", 0.95), _chunk("a5", 0.6)]
    merged = merge_retrieval_chunks(primary, secondary, final_k=3)
    assert len(merged) == 3
    assert {c["doc_id"] for c in merged} == {"a4", "a1", "a2"}
