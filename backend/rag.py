import json
import logging
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from pathlib import Path

# 避免 Windows 终端在加载 embedding 模型时 tqdm 进度条卡住
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import INDEX_STATS_FILE, settings
from inference_device import resolve_inference_device
from fetcher import fetch_law_raw, load_laws_config, parse_law
from context import filter_relevant_history
from classifier import is_legal_question
from llm import ask_llm, ask_llm_general
from parser import Article, dedupe_articles
from retrieval.bm25 import bm25_search, build_bm25_index, warmup_bm25
from retrieval.fusion import (
    build_rrf_rerank_pool,
    chunk_doc_id,
    rrf_merge_paths,
    rrf_select_topk,
)
from retrieval.rerank import (
    build_concat_search_query,
    build_rerank_queries,
    build_rerank_query,
    build_rerank_query_weights,
    rerank_hits,
    rerank_pool_fusion_mode,
    warmup_reranker,
)

_collection = None
_embedder: SentenceTransformer | None = None
_warmup_lock = threading.Lock()
_warmup_started = False
_ready = False
_ready_event = threading.Event()
logger = logging.getLogger(__name__)

_profile_enabled = False
_profile_ms: dict[str, float] = {}
_profile_session: ContextVar[dict[str, float] | None] = ContextVar("retrieval_profile_session", default=None)


def enable_retrieval_profile(enabled: bool = True) -> None:
    global _profile_enabled, _profile_ms
    _profile_enabled = enabled
    _profile_ms = {}


def get_retrieval_profile() -> dict[str, float]:
    session = _profile_session.get()
    if session is not None:
        return dict(session)
    return dict(_profile_ms)


def _profile_is_active() -> bool:
    return _profile_session.get() is not None or _profile_enabled


def _profile_add(name: str, elapsed_ms: float) -> None:
    session = _profile_session.get()
    if session is not None:
        session[name] = session.get(name, 0.0) + elapsed_ms
        return
    if _profile_enabled:
        _profile_ms[name] = _profile_ms.get(name, 0.0) + elapsed_ms


@contextmanager
def retrieval_profile_session():
    """按请求隔离检索 profiling（Trace 用，不影响 compare_rag 全局开关）。"""
    ms: dict[str, float] = {}
    token: Token = _profile_session.set(ms)
    try:
        yield ms
    finally:
        _profile_session.reset(token)


class _ProfileSpan:
    def __init__(self, name: str):
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        if _profile_is_active():
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if _profile_is_active():
            _profile_add(self.name, (time.perf_counter() - self.t0) * 1000)


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        device = resolve_inference_device(settings.inference_device)
        kwargs: dict = {"device": device}
        if settings.embedding_local_only:
            kwargs["local_files_only"] = True
        try:
            _embedder = SentenceTransformer(settings.embedding_model, **kwargs)
        except Exception as exc:
            if settings.embedding_local_only:
                logger.warning("本地模型加载失败，尝试在线下载: %s", exc)
                _embedder = SentenceTransformer(settings.embedding_model, device=device)
            else:
                raise
        logger.info("Embedding 模型已加载到 %s", device)
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name="legal_articles",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def is_ready() -> bool:
    return _ready


def wait_until_ready(timeout: float = 180) -> bool:
    """等待后台预热完成。"""
    if _ready:
        return True
    warmup()
    return _ready_event.wait(timeout=timeout)


def save_index_stats(counts: dict[str, int]) -> None:
    INDEX_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_STATS_FILE.write_text(
        json.dumps(counts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_index_stats() -> dict[str, int]:
    if not INDEX_STATS_FILE.exists():
        return {}
    try:
        return json.loads(INDEX_STATS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure_ready(timeout: float = 180) -> None:
    """等待后台预热完成；超时则抛出明确错误。"""
    if not wait_until_ready(timeout):
        raise RuntimeError("RAG 组件仍在加载中，请稍候几秒后重试")


def build_index() -> dict[str, int]:
    articles: list[Article] = []
    for law in load_laws_config():
        raw = fetch_law_raw(law)
        parsed = parse_law(law, raw)
        articles.extend(dedupe_articles(parsed))

    if not articles:
        raise RuntimeError("未解析到任何法条")

    collection = get_collection()
    if collection.count() > 0:
        existing = collection.get()
        collection.delete(ids=existing["ids"])

    embedder = get_embedder()
    ids = [a.doc_id for a in articles]
    documents = [a.embed_text() for a in articles]
    embeddings = embedder.encode(documents, normalize_embeddings=True).tolist()
    metadatas = [
        {
            "law_id": a.law_id,
            "law_name": a.law_name,
            "article_no": a.article_no,
            "hierarchy": a.hierarchy,
            "text": a.text,
            "source_url": a.source_url,
        }
        for a in articles
    ]
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    counts: dict[str, int] = {}
    for law in load_laws_config():
        counts[law["id"]] = sum(1 for a in articles if a.law_id == law["id"])
    save_index_stats(counts)
    if settings.bm25_enabled:
        build_bm25_index(articles)
    global _ready
    _ready = True
    _ready_event.set()
    return counts


def _retrieve_candidates(
    query: str,
    law_filter: str | None,
    candidate_k: int,
) -> list[dict]:
    """向量检索候选法条，带 doc_id 与相似度 score。"""
    batches = _retrieve_candidates_batch([query], law_filter, candidate_k)
    return batches[0] if batches else []


def _retrieve_candidates_batch(
    queries: list[str],
    law_filter: str | None,
    candidate_k: int,
) -> list[list[dict]]:
    """批量向量检索；与多次调用 _retrieve_candidates 结果一致。"""
    _ensure_ready()
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("向量库为空，请先运行: python scripts/build_index.py")

    valid = [q for q in queries if q and q.strip()]
    if not valid:
        return [[] for _ in queries]

    embedder = get_embedder()
    with _ProfileSpan("embed_ms"):
        encoded = embedder.encode(valid, normalize_embeddings=True)
    where = {"law_id": law_filter} if law_filter else None

    per_query: list[list[dict]] = []
    chroma_t0 = time.perf_counter() if _profile_is_active() else 0.0
    for vec in encoded:
        query_vec = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=candidate_k,
            where=where,
            include=["metadatas", "distances"],
        )
        hits: list[dict] = []
        for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
            doc_id = chunk_doc_id(meta)
            hits.append({**meta, "doc_id": doc_id, "score": 1 - dist})
        per_query.append(hits)

    if _profile_is_active():
        _profile_add("chroma_ms", (time.perf_counter() - chroma_t0) * 1000)

    out: list[list[dict]] = []
    vi = 0
    for q in queries:
        if q and q.strip():
            out.append(per_query[vi])
            vi += 1
        else:
            out.append([])
    return out


def _fetch_k_for_retrieval(final_k: int, *, rerank: bool | None = None) -> int:
    use_rerank = settings.rerank_enabled if rerank is None else rerank
    k = final_k
    if use_rerank:
        k = max(k, settings.rerank_candidate_k, settings.rrf_pool_k)
    if settings.bm25_enabled:
        k = max(k, settings.bm25_candidate_k)
    if use_rerank or settings.bm25_enabled:
        k = max(k, settings.retrieve_candidate_k)
    return k


def _rewrite_column_chunks(
    rewrite_q: str,
    law_filter: str | None,
    fetch_k: int,
    final_k: int,
    *,
    rewrite_source: str = "baseline",
    rewrite_vector_hits: list[dict] | None = None,
) -> list[dict]:
    """与 retrieve(rewrite_q, top_k=final_k, rerank=True) 同口径；可复用已有向量路。"""
    if rewrite_vector_hits is None:
        rewrite_vector_hits = _retrieve_candidates(rewrite_q, law_filter, fetch_k)
    else:
        rewrite_vector_hits = list(rewrite_vector_hits)

    for h in rewrite_vector_hits:
        h["fusion"] = "vector"

    by_id: dict[str, dict] = {}
    for h in rewrite_vector_hits:
        _merge_hit_store(by_id, h)

    path_hits: list[list[dict]] = [rewrite_vector_hits]
    if settings.bm25_enabled:
        with _ProfileSpan("rewrite_col_bm25_ms"):
            bm25_hits = bm25_search(rewrite_q, law_filter, fetch_k)
        path_hits.append(bm25_hits)
        for h in bm25_hits:
            _merge_hit_store(by_id, h)

    # 与 retrieve(rewrite_q) 一致：精排上下文为改写 query 本身，非原问
    with _ProfileSpan("rewrite_col_rerank_ms"):
        chunks, _, _, _ = _resolve_hybrid_chunks(
            rewrite_q,
            path_hits,
            final_k,
            use_rerank=True,
            by_id=by_id,
            rewrite_q=None,
            rewrite_source=rewrite_source,
        )
    return chunks


def _fuse_rewrite_union(
    question: str,
    rewrite_q: str,
    rewrite_source: str,
    retrieval_ctx,
    hybrid_chunks: list[dict],
    final_k: int,
    *,
    law_filter: str | None = None,
    fetch_k: int | None = None,
    rewrite_vector_hits: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """改写单路 Top-K ∪ 混合 Top-K 并集，再精排；与评测改写列同口径。"""
    meta: dict = {
        "rewrite_union_size": 0,
        "union_rerank": False,
        "union_rerank_skipped": False,
        "rewrite_column_top": [],
    }
    if not rewrite_q:
        return hybrid_chunks, meta

    fk = fetch_k if fetch_k is not None else _fetch_k_for_retrieval(final_k, rerank=True)
    with _ProfileSpan("rewrite_col_ms"):
        rewrite_column = _rewrite_column_chunks(
            rewrite_q,
            law_filter,
            fk,
            final_k,
            rewrite_source="baseline",
            rewrite_vector_hits=rewrite_vector_hits,
        )
    meta["rewrite_column_top"] = [h["article_no"] for h in rewrite_column]

    rewrite_ids = {h["doc_id"] for h in rewrite_column}
    hybrid_ids = {h["doc_id"] for h in hybrid_chunks}
    if rewrite_ids <= hybrid_ids:
        meta["rewrite_union_size"] = len(hybrid_chunks)
        meta["union_rerank_skipped"] = True
        return hybrid_chunks, meta

    by_id: dict[str, dict] = {}
    for h in rewrite_column + hybrid_chunks:
        by_id[h["doc_id"]] = h
    union = list(by_id.values())
    meta["rewrite_union_size"] = len(union)

    if len(union) <= final_k:
        return union[:final_k], meta

    rerank_queries = build_rerank_queries(
        question,
        rewrite_q,
        source=rewrite_source,
        query_type=retrieval_ctx.query_type,
    )
    query_weights = build_rerank_query_weights(
        question,
        rewrite_q,
        source=rewrite_source,
        query_type=retrieval_ctx.query_type,
    )
    with _ProfileSpan("union_rerank_ms"):
        fused = rerank_hits(
            rerank_queries,
            union,
            final_k,
            enabled=True,
            query_type=retrieval_ctx.query_type,
            query_weights=query_weights,
        )
    meta["union_rerank"] = True
    return fused, meta


def _resolve_hybrid_chunks(
    question: str,
    path_hits: list[list[dict]],
    final_k: int,
    *,
    use_rerank: bool,
    rewrite_hits: list[dict] | None = None,
    base_hits: list[dict] | None = None,
    by_id: dict[str, dict] | None = None,
    rewrite_q: str | None = None,
    rewrite_source: str = "baseline",
    retrieval_ctx=None,
    law_filter: str | None = None,
    fetch_k: int | None = None,
    rewrite_vector_hits: list[dict] | None = None,
) -> tuple[list[dict], str, int, dict]:
    """混合检索：Cascade 池精排 + 改写列 union 精排。"""
    from query_rewrite import infer_retrieval_context

    if retrieval_ctx is None:
        retrieval_ctx = infer_retrieval_context(
            question, rewrite_q, source=rewrite_source
        )
    extra_meta = {
        "query_type": retrieval_ctx.query_type,
        "inferred_law_id": retrieval_ctx.inferred_law_id,
        "domain_confidence": retrieval_ctx.domain_confidence,
    }
    multi_path = len(path_hits) > 1

    if use_rerank:
        if multi_path:
            with _ProfileSpan("pool_build_ms"):
                pool = build_rrf_rerank_pool(
                    path_hits,
                    pool_k=settings.rrf_pool_k,
                    rrf_k=settings.rrf_k,
                    bm25_max_entries=settings.bm25_rrf_max_entries,
                    bm25_weight=settings.bm25_rrf_weight,
                    inferred_law_id=retrieval_ctx.inferred_law_id,
                    domain_confidence=retrieval_ctx.domain_confidence,
                )
        else:
            pool = list((by_id or {}).values())[: settings.rrf_pool_k]
        pool_size = len(pool)
        reserve_count = sum(1 for h in pool if h.get("pool_source") == "reserve")
        extra_meta["pool_reserve_count"] = reserve_count
        rerank_queries = build_rerank_queries(
            question,
            rewrite_q,
            source=rewrite_source,
            query_type=retrieval_ctx.query_type,
        )
        query_weights = build_rerank_query_weights(
            question,
            rewrite_q,
            source=rewrite_source,
            query_type=retrieval_ctx.query_type,
        )
        with _ProfileSpan("hybrid_rerank_ms"):
            chunks = rerank_hits(
                rerank_queries,
                pool,
                final_k,
                enabled=True,
                query_type=retrieval_ctx.query_type,
                query_weights=query_weights,
            )
        if multi_path and rewrite_q:
            chunks, union_meta = _fuse_rewrite_union(
                question,
                rewrite_q,
                rewrite_source,
                retrieval_ctx,
                chunks,
                final_k,
                law_filter=law_filter,
                fetch_k=fetch_k,
                rewrite_vector_hits=rewrite_vector_hits,
            )
            extra_meta.update(union_meta)
        fusion_mode = rerank_pool_fusion_mode(retrieval_ctx.query_type)
        return chunks, fusion_mode, pool_size, extra_meta

    if settings.bm25_enabled and multi_path:
        chunks = rrf_merge_paths(
            path_hits,
            final_k,
            rrf_k=settings.rrf_k,
            bm25_max_entries=settings.bm25_rrf_max_entries,
            bm25_weight=settings.bm25_rrf_weight,
        )
        return chunks, "rrf_hybrid", len(chunks), extra_meta

    if base_hits is not None and rewrite_hits is not None:
        fused = rrf_select_topk(
            base_hits,
            rewrite_hits,
            top_k=final_k,
            rrf_k=settings.rrf_k,
        )
        by_id_vec = {h["doc_id"]: h for h in base_hits + rewrite_hits}
        chunks = []
        for doc_id, rrf_score in fused[:final_k]:
            hit = by_id_vec.get(doc_id)
            if hit:
                chunks.append({**hit, "score": rrf_score, "fusion": "rrf"})
        return chunks, "rrf", len(chunks), extra_meta

    chunks = list((by_id or {}).values())[:final_k]
    return chunks, "vector", len(chunks), extra_meta


def _merge_hit_store(store: dict[str, dict], hit: dict) -> None:
    doc_id = hit["doc_id"]
    if doc_id not in store:
        store[doc_id] = hit
        return
    existing = store[doc_id]
    if hit.get("fusion") == "bm25":
        existing["bm25_score"] = hit["score"]
    else:
        store[doc_id] = {**hit, "bm25_score": existing.get("bm25_score")}


def _merge_bm25_for_queries(
    queries: list[str],
    law_filter: str | None,
    candidate_k: int,
) -> list[dict]:
    best: dict[str, dict] = {}
    for q in queries:
        if not q or not q.strip():
            continue
        for hit in bm25_search(q, law_filter, candidate_k):
            doc_id = hit["doc_id"]
            if doc_id not in best or hit["score"] > best[doc_id]["score"]:
                best[doc_id] = hit
    return sorted(best.values(), key=lambda h: h["score"], reverse=True)


def _hybrid_retrieve_single(
    query: str,
    law_filter: str | None,
    candidate_k: int,
) -> tuple[list[dict], dict[str, dict], list[list[dict]]]:
    """单 query：向量 + 可选 BM25，返回向量 hits、并集、RRF 用分路径列表。"""
    vector_hits = _retrieve_candidates(query, law_filter, candidate_k)
    for h in vector_hits:
        h["fusion"] = "vector"
    by_id: dict[str, dict] = {}
    for h in vector_hits:
        _merge_hit_store(by_id, h)

    path_hits: list[list[dict]] = [vector_hits]
    if settings.bm25_enabled:
        bm25_hits = bm25_search(query, law_filter, candidate_k)
        path_hits.append(bm25_hits)
        for h in bm25_hits:
            _merge_hit_store(by_id, h)

    return vector_hits, by_id, path_hits


def _hybrid_retrieve_dual(
    base_q: str,
    rewrite_q: str,
    law_filter: str | None,
    candidate_k: int,
    *,
    concat_q: str | None = None,
) -> tuple[list[dict], list[dict], list[dict], dict[str, dict], list[list[dict]]]:
    """原问 + 改写 + 可选 concat 向量路 + BM25，供 RRF 融合。"""
    use_concat = (
        settings.concat_retrieval_enabled
        and concat_q
        and concat_q.strip() not in {base_q.strip(), rewrite_q.strip()}
    )

    batch_queries = [base_q, rewrite_q]
    if use_concat:
        batch_queries.append(concat_q)
    batch_hits = _retrieve_candidates_batch(batch_queries, law_filter, candidate_k)

    base_hits = batch_hits[0]
    rewrite_hits = batch_hits[1]
    concat_hits: list[dict] = batch_hits[2] if use_concat else []

    for h in base_hits:
        h["fusion"] = "vector"
    for h in rewrite_hits:
        h["fusion"] = "vector"

    by_id: dict[str, dict] = {}
    for h in base_hits + rewrite_hits:
        _merge_hit_store(by_id, h)

    path_hits: list[list[dict]] = [base_hits, rewrite_hits]

    if use_concat:
        for h in concat_hits:
            h["fusion"] = "concat_vector"
        path_hits.append(concat_hits)
        for h in concat_hits:
            _merge_hit_store(by_id, h)

    if settings.bm25_enabled:
        bm25_queries = [base_q, rewrite_q]
        if use_concat and concat_q:
            bm25_queries.append(concat_q)
        with _ProfileSpan("dual_bm25_ms"):
            bm25_hits = _merge_bm25_for_queries(bm25_queries, law_filter, candidate_k)
        path_hits.append(bm25_hits)
        for h in bm25_hits:
            _merge_hit_store(by_id, h)

    return base_hits, rewrite_hits, concat_hits, by_id, path_hits


def _finalize_hits(
    question: str,
    hits: list[dict],
    final_k: int,
    *,
    rerank: bool | None = None,
) -> list[dict]:
    if settings.rerank_enabled if rerank is None else rerank:
        return rerank_hits(question, hits, final_k, enabled=True)
    return hits[:final_k]


def retrieve(
    question: str,
    law_filter: str | None = None,
    top_k: int | None = None,
    *,
    rerank: bool | None = None,
) -> list[dict]:
    final_k = top_k or settings.top_k
    fetch_k = _fetch_k_for_retrieval(final_k, rerank=rerank)
    _, by_id, path_hits = _hybrid_retrieve_single(question, law_filter, fetch_k)
    use_rerank = settings.rerank_enabled if rerank is None else rerank
    chunks, _, _, _ = _resolve_hybrid_chunks(
        question,
        path_hits,
        final_k,
        use_rerank=use_rerank,
        by_id=by_id,
    )
    return chunks


def _attach_profile_to_meta(meta: dict, profile_ms: dict[str, float] | None) -> dict:
    out = dict(meta)
    if profile_ms is not None:
        out["profile_ms"] = dict(profile_ms)
    elif _profile_enabled:
        out["profile_ms"] = get_retrieval_profile()
    return out


def build_retrieve_trace_output(citations: list[dict], meta: dict, chunks: list[dict] | None = None) -> dict:
    output: dict = {
        "article_count": len(citations),
        "articles": [f"《{c['law_name']}》{c['article_no']}" for c in citations],
        "fusion_mode": meta.get("fusion_mode"),
        "search_query": meta.get("search_query"),
        "query_source": meta.get("query_source"),
    }
    if meta.get("profile_ms"):
        output["profile_ms"] = meta["profile_ms"]
    gap = meta.get("rerank_gap_truncate")
    if gap is None and chunks:
        gap = chunks[0].get("rerank_gap_truncate") if chunks else None
    if gap:
        output["rerank_gap_truncate"] = gap
    return output


def retrieve_fusion(
    question: str,
    history: list[dict] | None = None,
    law_filter: str | None = None,
    top_k: int | None = None,
    rewrite: bool | None = None,
    rerank: bool | None = None,
    *,
    profile: bool = False,
) -> tuple[list[dict], dict]:
    """Cascade 混合检索：多路召回 + 池精排 + 改写列 union 精排。"""
    profile_ctx = retrieval_profile_session() if profile else nullcontext()
    with profile_ctx as profile_ms:
        return _retrieve_fusion_impl(
            question,
            history,
            law_filter,
            top_k,
            rewrite,
            rerank,
            profile_ms=profile_ms if profile else None,
        )


def _retrieve_fusion_impl(
    question: str,
    history: list[dict] | None,
    law_filter: str | None,
    top_k: int | None,
    rewrite: bool | None,
    rerank: bool | None,
    *,
    profile_ms: dict[str, float] | None,
) -> tuple[list[dict], dict]:
    final_k = top_k or settings.top_k
    use_rerank = settings.rerank_enabled if rerank is None else rerank
    base_q = build_retrieval_query(question, history)
    use_rewrite = settings.query_rewrite_enabled if rewrite is None else rewrite
    hybrid_meta = {
        "retrieval": "cascade_union",
        "rerank": use_rerank,
        "bm25": settings.bm25_enabled,
        "bm25_rrf_max_entries": settings.bm25_rrf_max_entries,
        "bm25_rrf_weight": settings.bm25_rrf_weight,
        "rrf_pool_k": settings.rrf_pool_k,
        "path_reserve_vector_top": settings.path_reserve_vector_top,
        "path_reserve_bm25_top": settings.path_reserve_bm25_top,
        "domain_rrf_boost": settings.domain_rrf_boost,
        "concat_retrieval": settings.concat_retrieval_enabled,
        "concat_rrf_weight": settings.concat_rrf_weight,
    }

    if not use_rewrite:
        search_q, source = build_search_query(question, history, rewrite=use_rewrite)
        fetch_k = _fetch_k_for_retrieval(final_k, rerank=use_rerank)
        _, by_id, path_hits = _hybrid_retrieve_single(search_q, law_filter, fetch_k)
        chunks, fusion_mode, pool_size, cascade_meta = _resolve_hybrid_chunks(
            question,
            path_hits,
            final_k,
            use_rerank=use_rerank,
            by_id=by_id,
            rewrite_q=search_q if source not in ("baseline", "article_lookup") else None,
            rewrite_source=source,
        )
        rerank_query = build_rerank_query(
            question,
            search_q if source not in ("baseline", "article_lookup") else None,
            source=source,
        )
        rerank_queries = build_rerank_queries(
            question,
            search_q if source not in ("baseline", "article_lookup") else None,
            source=source,
        )
        return chunks, _attach_profile_to_meta(
            {
                "baseline_query": base_q,
                "rewrite_query": search_q,
                "search_query": search_q,
                "rerank_query": rerank_query,
                "rerank_queries": rerank_queries,
                "query_source": source,
                "rewrite_mode": settings.query_rewrite_mode,
                "fusion": fusion_mode not in {"vector"},
                "fusion_mode": fusion_mode,
                "rrf_pool_size": pool_size,
                **cascade_meta,
                **hybrid_meta,
            },
            profile_ms,
        )

    from query_rewrite import infer_retrieval_context, rewrite_for_search

    with _ProfileSpan("rewrite_api_ms"):
        rewrite_q, source, elements = rewrite_for_search(question, history)
    retrieval_ctx = infer_retrieval_context(
        question, rewrite_q, source=source, elements=elements
    )
    rewrite_meta: dict = {
        "rewrite_mode": settings.query_rewrite_mode,
        "rewrite_source": source,
    }
    if elements is not None:
        rewrite_meta["legal_elements"] = elements.to_dict()

    fetch_k = _fetch_k_for_retrieval(final_k, rerank=use_rerank)

    if source in ("baseline", "article_lookup") or not rewrite_q or rewrite_q.strip() == base_q.strip():
        _, by_id, path_hits = _hybrid_retrieve_single(base_q, law_filter, fetch_k)
        chunks, fusion_mode, pool_size, cascade_meta = _resolve_hybrid_chunks(
            question,
            path_hits,
            final_k,
            use_rerank=use_rerank,
            by_id=by_id,
            rewrite_source=source,
            retrieval_ctx=retrieval_ctx,
        )
        return chunks, _attach_profile_to_meta(
            {
                "baseline_query": base_q,
                "rewrite_query": rewrite_q or base_q,
                "search_query": base_q,
                "rerank_query": build_rerank_query(
                    question, None, source=source, query_type=retrieval_ctx.query_type
                ),
                "rerank_queries": build_rerank_queries(
                    question, None, source=source, query_type=retrieval_ctx.query_type
                ),
                "query_source": source,
                "fusion": fusion_mode not in {"vector"},
                "fusion_mode": fusion_mode,
                "rrf_pool_size": pool_size,
                **cascade_meta,
                **rewrite_meta,
                **hybrid_meta,
            },
            profile_ms,
        )

    concat_q = build_concat_search_query(question, rewrite_q, source=source)
    with _ProfileSpan("dual_retrieve_ms"):
        base_hits, rewrite_hits, concat_hits, by_id, path_hits = _hybrid_retrieve_dual(
            base_q,
            rewrite_q,
            law_filter,
            fetch_k,
            concat_q=concat_q,
        )

    rewrite_path_top = [h["article_no"] for h in rewrite_hits[:final_k]]
    baseline_path_top = [h["article_no"] for h in base_hits[:final_k]]
    concat_path_top = [h["article_no"] for h in concat_hits[:final_k]]

    chunks, fusion_mode, pool_size, cascade_meta = _resolve_hybrid_chunks(
        question,
        path_hits,
        final_k,
        use_rerank=use_rerank,
        rewrite_hits=rewrite_hits,
        base_hits=base_hits,
        by_id=by_id,
        rewrite_q=rewrite_q,
        rewrite_source=source,
        retrieval_ctx=retrieval_ctx,
        law_filter=law_filter,
        fetch_k=fetch_k,
        rewrite_vector_hits=rewrite_hits,
    )

    meta = _attach_profile_to_meta(
        {
            "baseline_query": base_q,
            "rewrite_query": rewrite_q,
            "search_query": rewrite_q,
            "rerank_query": build_rerank_query(
                question, rewrite_q, source=source, query_type=retrieval_ctx.query_type
            ),
            "rerank_queries": build_rerank_queries(
                question, rewrite_q, source=source, query_type=retrieval_ctx.query_type
            ),
            "query_source": fusion_mode,
            "fusion": fusion_mode not in {"vector"},
            "fusion_mode": fusion_mode,
            "rrf_pool_size": pool_size,
            "rewrite_path_top": rewrite_path_top,
            "baseline_path_top": baseline_path_top,
            "concat_query": concat_q or "",
            "concat_path_top": concat_path_top,
            **cascade_meta,
            **rewrite_meta,
            **hybrid_meta,
        },
        profile_ms,
    )
    return chunks, meta


def format_citations(chunks: list[dict]) -> list[dict]:
    return [
        {
            "law_name": c["law_name"],
            "article_no": c["article_no"],
            "hierarchy": c.get("hierarchy", ""),
            "text": c["text"],
            "source_url": c.get("source_url", ""),
        }
        for c in chunks
    ]


def build_retrieval_query(question: str, history: list[dict] | None = None) -> str:
    """结合相关对话历史构造检索 query（baseline，无 LLM 改写）。"""
    if not history:
        return question
    prev_users = [h["content"] for h in history if h.get("role") == "user"]
    if prev_users:
        return f"{prev_users[-1]} {question}"
    return question


def build_search_query(
    question: str,
    history: list[dict] | None = None,
    *,
    rewrite: bool | None = None,
) -> tuple[str, str]:
    """构造用于检索的 query。

    返回 (search_query, source)。
    source: two_stage | rewrite | baseline | article_lookup
    """
    from query_rewrite import is_article_lookup, rewrite_for_search

    base = build_retrieval_query(question, history)

    if is_article_lookup(question):
        return base, "article_lookup"

    use_rewrite = settings.query_rewrite_enabled if rewrite is None else rewrite
    if not use_rewrite:
        return base, "baseline"

    rewritten, source, _ = rewrite_for_search(question, history)
    if rewritten:
        if rewritten.strip() != base.strip():
            return rewritten.strip(), source
        return rewritten.strip(), "baseline"

    return base, "baseline"


def _resolve_history(question: str, history: list[dict] | None) -> list[dict]:
    return filter_relevant_history(question, history or [])


def answer_question(
    question: str,
    law_filter: str | None = None,
    history: list[dict] | None = None,
    trace=None,
) -> dict:
    from config import DISCLAIMER

    t0 = time.perf_counter()
    relevant = _resolve_history(question, history)
    if trace is not None:
        trace.step(
            "context_filter",
            (time.perf_counter() - t0) * 1000,
            {"context_turns": len(relevant)},
        )

    t0 = time.perf_counter()
    legal = is_legal_question(question, relevant or None)
    if trace is not None:
        trace.step("classify", (time.perf_counter() - t0) * 1000, {"is_legal": legal})

    if legal:
        t0 = time.perf_counter()
        chunks, meta = retrieve_fusion(
            question,
            relevant,
            law_filter=law_filter,
            profile=trace is not None and settings.trace_enabled,
        )
        citations = format_citations(chunks)
        if trace is not None:
            trace.step(
                "retrieve",
                (time.perf_counter() - t0) * 1000,
                build_retrieve_trace_output(citations, meta, chunks),
            )
        t0 = time.perf_counter()
        answer = ask_llm(question, chunks, relevant)
        if trace is not None:
            trace.step(
                "generate",
                (time.perf_counter() - t0) * 1000,
                {"answer_chars": len(answer)},
            )
        from verify.repair import verify_and_repair

        repair = verify_and_repair(
            answer,
            chunks,
            question=question,
            history=relevant,
            trace=trace,
        )
        answer = repair.answer
        citation_verified = repair.citation_verified
    else:
        t0 = time.perf_counter()
        answer = ask_llm_general(question, relevant)
        citations = []
        citation_verified = True
        if trace is not None:
            trace.step(
                "generate",
                (time.perf_counter() - t0) * 1000,
                {"answer_chars": len(answer)},
            )

    return {
        "answer": answer,
        "citations": citations,
        "disclaimer": DISCLAIMER,
        "is_legal": legal,
        "citation_verified": citation_verified if legal else True,
    }


def prepare_answer(
    question: str,
    law_filter: str | None = None,
    history: list[dict] | None = None,
    trace=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    relevant = _resolve_history(question, history)
    t0 = time.perf_counter()
    chunks, meta = retrieve_fusion(
        question,
        relevant,
        law_filter=law_filter,
        profile=trace is not None and settings.trace_enabled,
    )
    citations = format_citations(chunks)
    if trace is not None:
        trace.step(
            "retrieve",
            (time.perf_counter() - t0) * 1000,
            build_retrieve_trace_output(citations, meta, chunks),
        )
    return chunks, citations, relevant


def _warmup_worker() -> None:
    global _ready
    try:
        collection = get_collection()
        get_embedder()
        warmup_reranker()
        warmup_bm25()
        if collection.count() == 0:
            logger.warning("向量库为空，请运行 python scripts/build_index.py 构建索引")
        _ready = True
        _ready_event.set()
        logger.info("RAG 组件预热完成")
    except Exception as exc:
        logger.warning("预热失败，将在首次请求时重试: %s", exc)


def warmup() -> None:
    """后台预加载向量库与 embedding 模型，不阻塞 HTTP 服务启动。"""
    global _warmup_started
    with _warmup_lock:
        if _warmup_started:
            return
        _warmup_started = True
        threading.Thread(target=_warmup_worker, daemon=True, name="rag-warmup").start()


def list_laws() -> list[dict]:
    """读取法律列表；优先使用 index_stats.json，避免启动时查询 ChromaDB。"""
    laws = load_laws_config()
    stats = load_index_stats()
    if stats:
        return [
            {
                "id": law["id"],
                "name": law["name"],
                "source_url": law["source_url"],
                "article_count": stats.get(law["id"], 0),
            }
            for law in laws
        ]

    if not is_ready():
        return [
            {
                "id": law["id"],
                "name": law["name"],
                "source_url": law["source_url"],
                "article_count": 0,
            }
            for law in laws
        ]

    try:
        collection = get_collection()
        result = []
        for law in laws:
            data = collection.get(where={"law_id": law["id"]}, include=[])
            result.append(
                {
                    "id": law["id"],
                    "name": law["name"],
                    "source_url": law["source_url"],
                    "article_count": len(data["ids"]),
                }
            )
        save_index_stats({item["id"]: item["article_count"] for item in result})
        return result
    except Exception:
        return [
            {
                "id": law["id"],
                "name": law["name"],
                "source_url": law["source_url"],
                "article_count": 0,
            }
            for law in laws
        ]
