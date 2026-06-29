"""双路检索 RRF 融合工具。"""

from __future__ import annotations

import re

from config import settings


def chunk_doc_id(meta: dict) -> str:
    """与 parser.Article.doc_id 规则一致。"""
    safe = re.sub(r"[^\w]", "_", meta["article_no"])
    return f"{meta['law_id']}_{safe}"


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
    *,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion，返回 (doc_id, score) 降序列表。

    weights 与 ranked_lists 等长；BM25 路可设 <1 以降低对池子的影响。
    """
    scores: dict[str, float] = {}
    for i, ranked in enumerate(ranked_lists):
        w = 1.0 if not weights else weights[i]
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _path_rrf_weight(hits: list[dict]) -> float:
    """按路径类型返回 RRF 权重。"""
    if not hits:
        return 1.0
    fusion = hits[0].get("fusion", "vector")
    if fusion == "bm25":
        return settings.bm25_rrf_weight
    if fusion == "concat_vector":
        return settings.concat_rrf_weight
    return 1.0


def _prepare_rrf_inputs(
    path_hits: list[list[dict]],
    *,
    bm25_max_entries: int,
    bm25_weight: float,
) -> tuple[list[list[str]], list[float], dict[str, dict]]:
    """拆分多路检索；BM25 限条数；concat 路可加权。"""
    ranked_lists: list[list[str]] = []
    weights: list[float] = []
    by_id: dict[str, dict] = {}

    for hits in path_hits:
        if not hits:
            continue
        for h in hits:
            by_id[h["doc_id"]] = h
        if hits[0].get("fusion") == "bm25":
            capped = hits[: max(0, bm25_max_entries)]
            if capped:
                ranked_lists.append([h["doc_id"] for h in capped])
                weights.append(bm25_weight)
        else:
            ranked_lists.append([h["doc_id"] for h in hits])
            weights.append(_path_rrf_weight(hits))

    return ranked_lists, weights, by_id


def rrf_select_topk(
    base_hits: list[dict],
    rewrite_hits: list[dict],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """双路 RRF 融合选 Top-K（仅向量路，改写路优先补位）。"""
    rewrite_top = rewrite_hits[:top_k]
    base_top = base_hits[:top_k]
    base_ids = [h["doc_id"] for h in base_top]
    rewrite_ids = [h["doc_id"] for h in rewrite_top]
    scores = dict(rrf_fuse([base_ids, rewrite_ids], k=rrf_k))

    selected: list[str] = []
    seen: set[str] = set()
    for hit in rewrite_top:
        doc_id = hit["doc_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            selected.append(doc_id)
    for hit in base_top:
        if len(selected) >= top_k:
            break
        doc_id = hit["doc_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            selected.append(doc_id)

    return [(doc_id, scores.get(doc_id, 0.0)) for doc_id in selected[:top_k]]


def rrf_merge_paths(
    path_hits: list[list[dict]],
    top_k: int,
    *,
    rrf_k: int = 60,
    bm25_max_entries: int = 5,
    bm25_weight: float = 0.5,
) -> list[dict]:
    """多路 RRF 融合取 Top-K（BM25 限条数 + 降权）。"""
    ranked_lists, weights, by_id = _prepare_rrf_inputs(
        path_hits,
        bm25_max_entries=bm25_max_entries,
        bm25_weight=bm25_weight,
    )
    if not ranked_lists:
        return []

    fused = rrf_fuse(ranked_lists, k=rrf_k, weights=weights)
    return [
        {**by_id[doc_id], "score": score, "fusion": "rrf"}
        for doc_id, score in fused[:top_k]
        if doc_id in by_id
    ]


def build_rrf_rerank_pool(
    path_hits: list[list[dict]],
    *,
    pool_k: int,
    rrf_k: int = 60,
    bm25_max_entries: int = 5,
    bm25_weight: float = 0.5,
    inferred_law_id: str | None = None,
    domain_confidence: float = 0.0,
) -> list[dict]:
    """Cascade 建池：路径保底 ∪ RRF 填满（可选 domain 软加权）。"""
    ranked_lists, weights, by_id = _prepare_rrf_inputs(
        path_hits,
        bm25_max_entries=bm25_max_entries,
        bm25_weight=bm25_weight,
    )
    if not ranked_lists:
        return []

    pool: list[dict] = []
    seen: set[str] = set()

    def _append_hit(hit: dict, *, source: str, score: float = 0.0) -> None:
        doc_id = hit["doc_id"]
        if doc_id in seen:
            return
        seen.add(doc_id)
        pool.append({**hit, "score": score, "fusion": "rrf_pool", "pool_source": source})

    # Phase 1: 每路保底
    vec_reserve = max(0, settings.path_reserve_vector_top)
    bm25_reserve = max(0, settings.path_reserve_bm25_top)
    for hits in path_hits:
        if not hits:
            continue
        if hits[0].get("fusion") == "bm25":
            for h in hits[:bm25_reserve]:
                _append_hit(h, source="reserve")
        elif vec_reserve > 0:
            for h in hits[:vec_reserve]:
                _append_hit(h, source="reserve")

    # Phase 2: RRF 填满
    fused = rrf_fuse(ranked_lists, k=rrf_k, weights=weights)
    boost = (
        settings.domain_rrf_boost
        if inferred_law_id
        and domain_confidence >= settings.domain_boost_min_confidence
        else 1.0
    )

    for doc_id, score in fused:
        if len(pool) >= pool_k:
            break
        hit = by_id.get(doc_id)
        if not hit or doc_id in seen:
            continue
        if boost > 1.0 and hit.get("law_id") == inferred_law_id:
            score *= boost
        seen.add(doc_id)
        pool.append({
            **hit,
            "score": score,
            "fusion": "rrf_pool",
            "pool_source": "rrf",
        })

    return pool[:pool_k]
