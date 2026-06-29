"""BM25 稀疏检索：中文 jieba 分词 + rank_bm25。"""

from __future__ import annotations

import json
import logging
import pickle
import re
import threading
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from config import BM25_INDEX_DIR, BM25_INDEX_FILE, settings
from parser import Article
from retrieval.fusion import chunk_doc_id

logger = logging.getLogger(__name__)

_index_lock = threading.Lock()
_index_state: dict | None = None

# 检索时过滤单字噪声，保留条号中的汉字与数字
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff0-9]+")


def tokenize(text: str) -> list[str]:
    """法条检索分词：jieba 切分 + 过滤过短 token。"""
    tokens: list[str] = []
    for word in jieba.cut_for_search(text):
        word = word.strip()
        if len(word) < 2 and not word.isdigit():
            continue
        if _TOKEN_RE.fullmatch(word) or word.isdigit():
            tokens.append(word)
    return tokens or list(jieba.cut(text))


def _article_search_text(article: Article) -> str:
    return article.embed_text()


def _meta_from_article(article: Article) -> dict:
    return {
        "law_id": article.law_id,
        "law_name": article.law_name,
        "article_no": article.article_no,
        "hierarchy": article.hierarchy,
        "text": article.text,
        "source_url": article.source_url,
    }


def build_bm25_index(articles: list[Article]) -> int:
    """根据法条列表构建 BM25 索引并持久化。"""
    if not articles:
        raise RuntimeError("BM25 建库失败：无法条")

    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    corpus_tokens = [tokenize(_article_search_text(a)) for a in articles]
    model = BM25Okapi(corpus_tokens)

    doc_ids = [a.doc_id for a in articles]
    metadatas = [_meta_from_article(a) for a in articles]

    payload = {
        "version": 1,
        "corpus_tokens": corpus_tokens,
        "doc_ids": doc_ids,
        "metadatas": metadatas,
    }
    BM25_INDEX_FILE.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    stats = {
        "article_count": len(articles),
        "avg_tokens": round(sum(len(t) for t in corpus_tokens) / len(corpus_tokens), 2),
    }
    (BM25_INDEX_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    global _index_state
    with _index_lock:
        _index_state = {
            "model": model,
            "doc_ids": doc_ids,
            "metadatas": metadatas,
        }

    logger.info("BM25 索引构建完成：%d 条法条", len(articles))
    return len(articles)


def _load_index() -> dict | None:
    global _index_state
    if _index_state is not None:
        return _index_state

    with _index_lock:
        if _index_state is not None:
            return _index_state
        if not BM25_INDEX_FILE.exists():
            return None
        payload = pickle.loads(BM25_INDEX_FILE.read_bytes())
        model = BM25Okapi(payload["corpus_tokens"])
        _index_state = {
            "model": model,
            "doc_ids": payload["doc_ids"],
            "metadatas": payload["metadatas"],
        }
        return _index_state


def is_bm25_ready() -> bool:
    return _load_index() is not None


def warmup_bm25() -> None:
    if settings.bm25_enabled:
        _load_index()


def bm25_search(
    query: str,
    law_filter: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """BM25 检索，返回与向量检索一致的 hit 结构。"""
    if not settings.bm25_enabled:
        return []

    state = _load_index()
    if state is None:
        logger.warning("BM25 索引不存在，请运行 build_index.py；已跳过 BM25 检索")
        return []

    k = top_k or settings.top_k
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    model: BM25Okapi = state["model"]
    scores = model.get_scores(query_tokens)

    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    hits: list[dict] = []
    for idx, score in ranked:
        if len(hits) >= k:
            break
        meta = state["metadatas"][idx]
        if law_filter and meta.get("law_id") != law_filter:
            continue
        hits.append(
            {
                **meta,
                "doc_id": state["doc_ids"][idx],
                "score": float(score),
                "fusion": "bm25",
            }
        )
    return hits
