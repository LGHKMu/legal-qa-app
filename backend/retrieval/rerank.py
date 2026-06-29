"""Cross-Encoder 重排序：对向量检索候选做法条级精排，可选 MMR 多样性。"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from config import DATA_DIR, settings
from inference_device import resolve_inference_device

_reranker = None
_reranker_lock = threading.Lock()
logger = logging.getLogger(__name__)

DEFAULT_LOCAL_RERANK_DIR = DATA_DIR / "models" / "bge-reranker-base"


def _is_valid_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    return (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()


def resolve_rerank_model_id() -> str:
    """解析 CrossEncoder 加载路径：本地目录 > HF 模型 ID。"""
    if settings.rerank_model_path:
        custom = Path(settings.rerank_model_path)
        if not custom.is_absolute():
            custom = Path(__file__).resolve().parent.parent / custom
        if _is_valid_model_dir(custom):
            return str(custom.resolve())
        logger.warning("RERANK_MODEL_PATH 无效或缺少权重: %s", custom)

    if _is_valid_model_dir(DEFAULT_LOCAL_RERANK_DIR):
        return str(DEFAULT_LOCAL_RERANK_DIR.resolve())

    return settings.rerank_model


def get_reranker():
    """懒加载 Cross-Encoder 模型（与 embedding 模型独立）。"""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                model_id = resolve_rerank_model_id()
                is_local = Path(model_id).is_dir()
                device = resolve_inference_device(settings.inference_device)
                kwargs: dict = {"device": device}
                if settings.rerank_local_only or is_local:
                    kwargs["local_files_only"] = True

                try:
                    _reranker = CrossEncoder(model_id, **kwargs)
                except Exception as exc:
                    if is_local:
                        raise RuntimeError(
                            f"本地 Reranker 加载失败: {model_id}。请运行 "
                            "python scripts/download_reranker.py"
                        ) from exc
                    if settings.rerank_local_only:
                        raise RuntimeError(
                            "未找到本地 Reranker，且 RERANK_LOCAL_ONLY=true。"
                            "请运行: python scripts/download_reranker.py\n"
                            "或设置 RERANK_MODEL_PATH=./data/models/bge-reranker-base"
                        ) from exc
                    logger.warning("在线加载 Reranker 失败: %s", exc)
                    raise
                logger.info("Reranker 加载完成: %s (device=%s)", model_id, device)
    return _reranker


_RERANK_INTENT_LABEL = "检索意图："
_RERANK_FACT_LABEL = "案情："


def _build_concat_rerank_query(orig: str, rw: str) -> str:
    """方案 B：检索意图 + 案情（截断）。"""
    prefix = f"{_RERANK_INTENT_LABEL}{rw}\n{_RERANK_FACT_LABEL}"
    max_chars = settings.rerank_query_max_chars
    budget = max_chars - len(prefix)
    if budget <= 0:
        intent_budget = max_chars - len(_RERANK_INTENT_LABEL)
        return f"{_RERANK_INTENT_LABEL}{rw[:max(0, intent_budget)]}"
    fact = orig if len(orig) <= budget else orig[:budget]
    return f"{prefix}{fact}"


def _rewrite_skipped(question: str, rewrite_q: str | None, *, source: str) -> bool:
    orig = question.strip()
    rw = (rewrite_q or "").strip()
    return source in ("article_lookup", "baseline") or not rw or rw == orig


def build_concat_search_query(
    question: str,
    rewrite_q: str | None = None,
    *,
    source: str = "baseline",
) -> str | None:
    """构造 concat 检索 query（与精排方案 B 同构）；无改写时返回 None。"""
    if _rewrite_skipped(question, rewrite_q, source=source):
        return None
    return _build_concat_rerank_query(question.strip(), (rewrite_q or "").strip())


def _selection_for_query_type(query_type: str | None) -> str:
    """案情/条号纯相关性；概念题组间多样化。"""
    if query_type in ("case", "statute"):
        return "plain"
    return "group_constrained"


def build_rerank_queries(
    question: str,
    rewrite_q: str | None = None,
    *,
    source: str = "baseline",
    query_type: str | None = None,
) -> list[str]:
    """按 query_type 构造精排 query。"""
    orig = question.strip()
    if _rewrite_skipped(question, rewrite_q, source=source):
        return [orig]

    rw = (rewrite_q or "").strip()
    concat_q = _build_concat_rerank_query(orig, rw)
    qt = query_type or "concept"
    if qt == "case":
        return [rw]
    if qt == "statute":
        return [orig]
    return list(dict.fromkeys([orig, rw, concat_q]))


def build_rerank_query_weights(
    question: str,
    rewrite_q: str | None = None,
    *,
    source: str = "baseline",
    query_type: str | None = None,
) -> list[float] | None:
    """概念题多 query 加权；单 query 返回 None。"""
    if _rewrite_skipped(question, rewrite_q, source=source):
        return None
    if (query_type or "concept") != "concept":
        return None
    queries = build_rerank_queries(
        question, rewrite_q, source=source, query_type=query_type
    )
    if len(queries) <= 1:
        return None
    orig = question.strip()
    rw = (rewrite_q or "").strip()
    concat_q = _build_concat_rerank_query(orig, rw)
    weights: list[float] = []
    for q in queries:
        if q == rw:
            weights.append(settings.rerank_weight_rewrite)
        elif q == concat_q:
            weights.append(settings.rerank_weight_concat)
        else:
            weights.append(settings.rerank_weight_orig)
    return weights


def build_rerank_query(
    question: str,
    rewrite_q: str | None = None,
    *,
    source: str = "baseline",
    query_type: str | None = None,
) -> str:
    """单条精排 query（展示用）；多 query 时返回 concat。"""
    queries = build_rerank_queries(
        question, rewrite_q, source=source, query_type=query_type
    )
    if len(queries) == 1:
        return queries[0]
    rw = (rewrite_q or "").strip()
    return _build_concat_rerank_query(question.strip(), rw)


def chunk_rerank_text(hit: dict) -> str:
    """构造与入库 embedding 一致的法条文本，供 Cross-Encoder 打分。"""
    parts = [f"《{hit['law_name']}》"]
    if hit.get("hierarchy"):
        parts.append(hit["hierarchy"])
    parts.append(hit["article_no"])
    parts.append(hit["text"])
    return "\n".join(parts)


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.ones(len(scores), dtype=np.float64)
    return (scores - lo) / (hi - lo)


def _encode_hit_embeddings(hits: list[dict]) -> np.ndarray:
    """用法条文本 embedding 估计文档间语义相似度（MMR 多样性项）。"""
    from rag import get_embedder

    embedder = get_embedder()
    texts = [chunk_rerank_text(h) for h in hits]
    return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _pairwise_diversity(i: int, j: int, embeddings: np.ndarray, hit_i: dict, hit_j: dict) -> float:
    """文档间多样性惩罚；默认同部法律不互斥（劳动法相邻条文可共存）。"""
    if (
        not settings.rerank_mmr_diversify_same_law
        and hit_i.get("law_id")
        and hit_i.get("law_id") == hit_j.get("law_id")
    ):
        return 0.0
    return float(np.dot(embeddings[i], embeddings[j]))


def _cluster_indices(
    indices: list[int],
    embeddings: np.ndarray,
    relevance_scores: list[float] | np.ndarray,
    *,
    threshold: float,
) -> list[list[int]]:
    """区内贪心聚类：高相关种子 + 相似度阈值扩展。"""
    remaining = set(indices)
    groups: list[list[int]] = []
    scores = np.asarray(relevance_scores, dtype=np.float64)

    while remaining:
        seed = max(remaining, key=lambda i: scores[i])
        group = [seed]
        remaining.remove(seed)
        changed = True
        while changed:
            changed = False
            for j in list(remaining):
                max_sim = max(float(np.dot(embeddings[j], embeddings[g])) for g in group)
                if max_sim >= threshold:
                    group.append(j)
                    remaining.remove(j)
                    changed = True
        groups.append(group)
    return groups


def build_issue_groups(
    hits: list[dict],
    embeddings: np.ndarray,
    relevance_scores: list[float] | np.ndarray,
    *,
    threshold: float | None = None,
) -> list[list[int]]:
    """按 law_id 分区，再按 embedding 相似度聚为议题组。"""
    thresh = threshold if threshold is not None else settings.rerank_group_cluster_threshold
    by_law: dict[str, list[int]] = {}
    for i, hit in enumerate(hits):
        law = hit.get("law_id") or "_"
        by_law.setdefault(law, []).append(i)

    groups: list[list[int]] = []
    for law_indices in by_law.values():
        if len(law_indices) == 1:
            groups.append(law_indices)
            continue
        groups.extend(
            _cluster_indices(law_indices, embeddings, relevance_scores, threshold=thresh)
        )
    return groups


def group_constrained_select_indices(
    hits: list[dict],
    relevance_scores: list[float] | np.ndarray,
    groups: list[list[int]],
    *,
    top_k: int,
    lambda_mult: float,
    embeddings: np.ndarray,
    seed_selected: list[int] | None = None,
) -> list[int]:
    """组内按相关性竞争；仅对不同议题组之间施加多样性惩罚。"""
    n = len(hits)
    if n == 0:
        return []
    k = min(top_k, n)

    rel = _minmax_normalize(np.asarray(relevance_scores, dtype=np.float64))
    index_to_group: dict[int, int] = {}
    for gid, group in enumerate(groups):
        for idx in group:
            index_to_group[idx] = gid

    selected: list[int] = list(seed_selected or [])
    candidates = set(range(n)) - set(selected)

    while len(selected) < k and candidates:
        best_idx = -1
        best_mmr = -float("inf")
        for i in candidates:
            if not selected:
                mmr = rel[i]
            else:
                max_div = 0.0
                gi = index_to_group.get(i, -1)
                for j in selected:
                    if index_to_group.get(j, -2) != gi:
                        max_div = max(
                            max_div,
                            _pairwise_diversity(i, j, embeddings, hits[i], hits[j]),
                        )
                mmr = lambda_mult * rel[i] - (1.0 - lambda_mult) * max_div
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected


def _predict_relevance_scores(
    queries: list[str],
    hits: list[dict],
    *,
    query_weights: list[float] | None = None,
) -> list[float]:
    """对候选法条打分；多 query 时加权求和或取 max。"""
    model = get_reranker()
    doc_texts = [chunk_rerank_text(h) for h in hits]
    if len(queries) == 1:
        pairs = [(queries[0], text) for text in doc_texts]
        return [float(s) for s in model.predict(pairs, show_progress_bar=False)]

    pairs = [(q, text) for q in queries for text in doc_texts]
    raw = model.predict(pairs, show_progress_bar=False)
    matrix = np.asarray(raw, dtype=np.float64).reshape(len(queries), len(hits))
    for i in range(matrix.shape[0]):
        matrix[i] = _minmax_normalize(matrix[i])

    if query_weights and len(query_weights) == len(queries):
        weights = np.asarray(query_weights, dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-9)
        return (matrix.T @ weights).tolist()

    default_w = np.asarray(
        [
            settings.rerank_weight_orig,
            settings.rerank_weight_rewrite,
            settings.rerank_weight_concat,
        ],
        dtype=np.float64,
    )[: len(queries)]
    default_w = default_w / max(default_w.sum(), 1e-9)
    return (matrix.T @ default_w).tolist()


def _hit_score(hit: dict) -> float:
    if "rerank_score" in hit:
        return float(hit["rerank_score"])
    return float(hit.get("score", 0.0))


def _truncate_reject_reason(
    top_score: float,
    prev_score: float,
    cur_score: float,
    *,
    min_relative: float,
    max_step_drop: float,
    check_step_drop: bool = True,
) -> str | None:
    """双条件截断判别；返回 None 表示可保留，否则为截断原因。

    保留当且仅当：
      ① cur >= top × α（始终）
      ② cur >= prev × (1 − γ)（仅当 check_step_drop=True 时检查）
    """
    if top_score <= 0:
        return "invalid_top_score"
    if cur_score < top_score * min_relative:
        return "relative_to_top"
    if check_step_drop and prev_score > 0 and cur_score < prev_score * (1.0 - max_step_drop):
        return "relative_step_drop"
    return None


def truncate_by_score_gap(
    hits: list[dict],
    *,
    max_k: int | None = None,
    min_k: int | None = None,
    min_relative: float | None = None,
    max_step_drop: float | None = None,
) -> list[dict]:
    """精排后按「相对首条保底 α + 相邻跌幅 γ」截断，不再凑满 top_k。

    - α（相对首条）：从第 2 条起始终检查，min_k 不能豁免；防止「与上一条差距小
      但与首条差距已过大」的条目（如第 4 条）仍被链式选入。
    - γ（相邻跌幅）：仅在已保留 min_k 条之后，才用相邻跌幅截断尾部。
    """
    if not hits:
        return []

    limit = max_k if max_k is not None else settings.top_k
    floor = min_k if min_k is not None else settings.rerank_gap_truncate_min
    alpha = min_relative if min_relative is not None else settings.rerank_truncate_min_relative
    gamma = max_step_drop if max_step_drop is not None else settings.rerank_truncate_max_step_drop

    if not settings.rerank_gap_truncate_enabled or len(hits) <= floor:
        return hits[:limit]

    ranked = sorted(hits, key=_hit_score, reverse=True)
    top_score = _hit_score(ranked[0])
    kept: list[dict] = [ranked[0]]
    cut_reason: str | None = None
    cut_relative_top: float | None = None
    cut_step_drop: float | None = None

    for nxt in ranked[1:]:
        if len(kept) >= limit:
            break
        prev_score = _hit_score(kept[-1])
        cur_score = _hit_score(nxt)
        reason = _truncate_reject_reason(
            top_score,
            prev_score,
            cur_score,
            min_relative=alpha,
            max_step_drop=gamma,
            check_step_drop=len(kept) >= floor,
        )
        if reason:
            cut_reason = reason
            cut_relative_top = cur_score / top_score if top_score else None
            cut_step_drop = (
                (prev_score - cur_score) / prev_score if prev_score > 0 else None
            )
            break

        kept.append(nxt)

    if len(kept) < len(ranked):
        meta = {
            "before": len(ranked),
            "after": len(kept),
            "rule": "cur>=top*α (always); cur>=prev*(1-γ) after min_k",
            "min_relative": alpha,
            "max_step_drop": gamma,
            "cut_reason": cut_reason,
            "cut_relative_top": cut_relative_top,
            "cut_step_drop": cut_step_drop,
        }
        kept[0] = {**kept[0], "rerank_gap_truncate": meta}

    return kept


def rerank_hits(
    query: str | list[str],
    hits: list[dict],
    top_k: int | None = None,
    *,
    enabled: bool | None = None,
    query_type: str | None = None,
    query_weights: list[float] | None = None,
) -> list[dict]:
    """Cascade 池精排：按 query_type 选条（plain / 议题组）。"""
    if not hits:
        return []

    k = top_k or settings.top_k
    use_rerank = settings.rerank_enabled if enabled is None else enabled
    if not use_rerank or len(hits) <= k:
        return truncate_by_score_gap(hits[:k], max_k=k)

    queries = [query] if isinstance(query, str) else [q for q in query if q and q.strip()]
    if not queries:
        return truncate_by_score_gap(hits[:k], max_k=k)

    score_list = _predict_relevance_scores(queries, hits, query_weights=query_weights)
    selection_mode = _selection_for_query_type(query_type)

    if selection_mode == "group_constrained" and k < len(hits):
        ranked_indices = sorted(range(len(hits)), key=lambda i: score_list[i], reverse=True)
        pure_lead = min(max(0, settings.rerank_mmr_pure_lead), k)
        seed = ranked_indices[:pure_lead]
        embeddings = _encode_hit_embeddings(hits)
        groups = build_issue_groups(hits, embeddings, score_list)
        picked = group_constrained_select_indices(
            hits,
            score_list,
            groups,
            top_k=k,
            lambda_mult=settings.rerank_mmr_lambda,
            embeddings=embeddings,
            seed_selected=seed,
        )
        result: list[dict] = []
        for rank, idx in enumerate(picked):
            hit = hits[idx]
            score = score_list[idx]
            fusion = hit.get("fusion", "rerank_group")
            if fusion in {"rrf_pool"}:
                fusion = "rerank_group"
            result.append({
                **hit,
                "score": score,
                "rerank_score": score,
                "mmr_rank": rank + 1,
                "fusion": fusion,
            })
        return truncate_by_score_gap(result, max_k=k)

    ranked = sorted(zip(hits, score_list), key=lambda item: item[1], reverse=True)
    result = []
    for hit, score in ranked[:k]:
        merged = {**hit, "score": score, "rerank_score": score}
        if "fusion" not in merged:
            merged["fusion"] = "rerank"
        result.append(merged)
    return truncate_by_score_gap(result, max_k=k)


def warmup_reranker() -> None:
    if settings.rerank_enabled:
        get_reranker()


def rerank_pool_fusion_mode(query_type: str | None = None) -> str:
    """混合检索 fusion_mode。"""
    if _selection_for_query_type(query_type) == "group_constrained":
        return "cascade_pool_rerank_group"
    return "cascade_pool_rerank"
