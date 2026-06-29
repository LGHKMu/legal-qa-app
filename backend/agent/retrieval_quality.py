"""案情检索质量评估与多轮检索结果合并。"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import settings


def _topic_relevance(question: str, chunks: list[dict]) -> tuple[bool, str]:
    from query_rewrite import topic_relevance_ok

    return topic_relevance_ok(question, chunks)


@dataclass
class RetrievalQuality:
    sufficient: bool
    reason: str = ""
    signals: dict = field(default_factory=dict)


def _chunk_score(chunk: dict) -> float | None:
    if chunk.get("rerank_score") is not None:
        return float(chunk["rerank_score"])
    if chunk.get("score") is not None:
        return float(chunk["score"])
    return None


def _chunk_key(chunk: dict) -> tuple:
    doc_id = chunk.get("doc_id")
    if doc_id:
        return ("doc_id", doc_id)
    return ("article", chunk.get("law_name", ""), chunk.get("article_no", ""))


def _distinct_law_ids(chunks: list[dict]) -> set[str]:
    return {c.get("law_id") or c.get("law_name") or "" for c in chunks if c}


def assess_retrieval_quality(chunks: list[dict], meta: dict | None, *, question: str = "") -> RetrievalQuality:
    """规则判断首轮检索是否足够支撑案情回答（不调 LLM）。"""
    meta = meta or {}
    if not chunks:
        return RetrievalQuality(
            sufficient=False,
            reason="empty_results",
            signals={"chunk_count": 0},
        )

    scores = [_chunk_score(c) for c in chunks]
    numeric = [s for s in scores if s is not None]
    signals: dict = {
        "chunk_count": len(chunks),
        "query_type": meta.get("query_type"),
        "domain_confidence": meta.get("domain_confidence"),
        "fusion_mode": meta.get("fusion_mode"),
    }

    reasons: list[str] = []

    if numeric:
        top = numeric[0]
        signals["top_score"] = round(top, 4)
        if top < settings.agent_case_retry_min_top_score:
            reasons.append("low_top_score")

        if len(numeric) >= 2:
            gap = top - numeric[1]
            signals["top_score_gap"] = round(gap, 4)
            if gap < settings.agent_case_retry_min_score_gap:
                reasons.append("low_score_gap")

    law_ids = _distinct_law_ids(chunks)
    signals["law_id_count"] = len(law_ids)
    if len(law_ids) > settings.agent_case_retry_max_law_ids and len(numeric) >= 3:
        spread = numeric[0] - numeric[2]
        signals["top3_spread"] = round(spread, 4)
        if spread < settings.agent_case_retry_min_score_gap * 2:
            reasons.append("dispersed_laws")

    domain_conf = meta.get("domain_confidence")
    if domain_conf is not None:
        signals["domain_confidence"] = float(domain_conf)
        if (
            meta.get("query_type") == "case"
            and float(domain_conf) < settings.agent_case_retry_min_domain_conf
        ):
            reasons.append("low_domain_confidence")

    gap_meta = meta.get("rerank_gap_truncate")
    if not gap_meta and chunks:
        gap_meta = chunks[0].get("rerank_gap_truncate")
    if gap_meta and len(chunks) <= settings.rerank_gap_truncate_min:
        kept = gap_meta.get("kept_count") or len(chunks)
        pool = gap_meta.get("pool_size") or meta.get("rrf_pool_size")
        signals["gap_truncate"] = gap_meta
        if pool and int(pool) >= settings.rrf_pool_k // 2 and kept <= settings.rerank_gap_truncate_min:
            if numeric and numeric[0] < settings.agent_case_retry_min_top_score:
                reasons.append("aggressive_gap_truncate")

    if question:
        topic_ok, topic_reason = _topic_relevance(question, chunks)
        signals["topic_relevant"] = topic_ok
        if not topic_ok:
            reasons.insert(0, topic_reason)

    sufficient = len(reasons) == 0
    return RetrievalQuality(
        sufficient=sufficient,
        reason=reasons[0] if reasons else "",
        signals=signals,
    )


def merge_retrieval_chunks(
    primary: list[dict],
    secondary: list[dict],
    *,
    final_k: int | None = None,
) -> list[dict]:
    """合并两轮检索：primary 优先保留顺序，按分数截断到 final_k。"""
    k = final_k or settings.top_k
    seen: set[tuple] = set()
    merged: list[dict] = []

    for chunk in primary:
        key = _chunk_key(chunk)
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    for chunk in secondary:
        key = _chunk_key(chunk)
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    if len(merged) <= k:
        return merged

    merged.sort(key=lambda c: _chunk_score(c) if _chunk_score(c) is not None else -1.0, reverse=True)
    return merged[:k]
