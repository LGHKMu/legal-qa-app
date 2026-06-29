"""RRF 融合与 query 拼接冒烟测试（纯本地）。"""

from __future__ import annotations

from query_rewrite import LegalElements, build_query_from_elements
from retrieval.fusion import rrf_select_topk


def test_rrf_preserves_rewrite_tail_doc() -> None:
    rewrite_hits = [{"doc_id": f"rw_{i}", "article_no": f"R{i}"} for i in range(5)]
    rewrite_hits[4]["doc_id"] = "target_43"
    rewrite_hits[4]["article_no"] = "第四十三条"
    base_hits = [{"doc_id": f"base_{i}", "article_no": f"B{i}"} for i in range(5)]

    fused = rrf_select_topk(base_hits, rewrite_hits, top_k=5, rrf_k=60)
    fused_ids = [doc_id for doc_id, _ in fused]
    assert "target_43" in fused_ids


def test_build_query_from_elements() -> None:
    elements = LegalElements(
        domains=["宪法", "劳动法"],
        query_keywords=["言论自由", "停职", "劳动者权利"],
        topics=["停职"],
    )
    q = build_query_from_elements(elements)
    assert "言论自由" in q
    assert "停职" in q
