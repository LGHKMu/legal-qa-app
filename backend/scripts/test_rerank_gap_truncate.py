"""rerank 截断单元测试（相对首条 α + 相邻跌幅 γ）。

用法:
  cd backend
  python scripts/test_rerank_gap_truncate.py
  python scripts/test_rerank_gap_truncate.py --live
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from retrieval.rerank import truncate_by_score_gap


def _hit(article_no: str, score: float) -> dict:
    return {
        "law_name": "中华人民共和国宪法",
        "article_no": article_no,
        "text": "…",
        "rerank_score": score,
    }


def test_gradual_ladder_drops_tail() -> None:
    """0.98, 0.76, 0.55, 0.43 — 仅保留 0.98（0.76/0.75 档亦剔除）。"""
    settings.rerank_gap_truncate_enabled = True
    settings.rerank_truncate_min_relative = 0.78
    settings.rerank_truncate_max_step_drop = 0.22
    hits = [
        _hit("a", 0.98),
        _hit("b", 0.76),
        _hit("c", 0.55),
        _hit("d", 0.43),
    ]
    kept = truncate_by_score_gap(hits, max_k=5, min_k=1)
    assert len(kept) == 1, [(_hit_score(h), h["article_no"]) for h in kept]
    assert kept[0]["article_no"] == "a"
    print("[OK] ladder 0.98..0.43 keeps only 0.98")


def _hit_score(h: dict) -> float:
    return float(h["rerank_score"])


def test_min_k_does_not_bypass_top_relative() -> None:
    """min_k=2 时，第 2 条若相对首条不足 α，仍应截断（不能靠 min_k 硬塞）。"""
    settings.rerank_gap_truncate_enabled = True
    settings.rerank_truncate_min_relative = 0.72
    settings.rerank_truncate_max_step_drop = 0.25
    hits = [
        _hit("a", 0.98),
        _hit("b", 0.60),
        _hit("c", 0.58),
    ]
    kept = truncate_by_score_gap(hits, max_k=5, min_k=2, min_relative=0.72)
    assert len(kept) == 1
    assert kept[0]["article_no"] == "a"
    print("[OK] min_k does not bypass relative_to_top")


def test_chain_small_steps_cut_by_top_floor() -> None:
    """第 4 条与第 3 条差距小，但与首条差距过大 → 被 α 截断。"""
    settings.rerank_gap_truncate_enabled = True
    hits = [
        _hit("a", 0.98),
        _hit("b", 0.90),
        _hit("c", 0.83),
        _hit("d", 0.76),
        _hit("e", 0.72),
    ]
    kept = truncate_by_score_gap(
        hits, max_k=5, min_k=2, min_relative=0.78, max_step_drop=0.22
    )
    assert len(kept) == 3
    assert [h["article_no"] for h in kept] == ["a", "b", "c"]
    assert kept[0]["rerank_gap_truncate"]["cut_reason"] == "relative_to_top"
    print("[OK] chain small steps cut by top floor at rank 4")


def test_cliff_after_top_two() -> None:
    hits = [
        _hit("第三十五条", 0.95),
        _hit("第五十一条", 0.92),
        _hit("第四十七条", 0.41),
        _hit("第三十七条", 0.39),
    ]
    kept = truncate_by_score_gap(hits, max_k=5, min_k=1)
    assert len(kept) == 2
    assert kept[0]["rerank_gap_truncate"]["cut_reason"] in (
        "relative_step_drop",
        "relative_to_top",
    )
    print("[OK] cliff after top-2 keeps 2")


def test_flat_scores_keeps_max_k() -> None:
    hits = [_hit(f"第{i}条", 0.90 - i * 0.02) for i in range(1, 6)]
    kept = truncate_by_score_gap(hits, max_k=5, min_k=1)
    assert len(kept) == 5
    print("[OK] flat scores keep max_k")


def test_disabled_keeps_all() -> None:
    settings.rerank_gap_truncate_enabled = False
    hits = [_hit("第三十五条", 0.98), _hit("第四十七条", 0.43)]
    kept = truncate_by_score_gap(hits, max_k=5)
    assert len(kept) == 2
    settings.rerank_gap_truncate_enabled = True
    print("[OK] disabled keeps all up to max_k")


def test_live_retrieve() -> None:
    from rag import retrieve_fusion, wait_until_ready

    if not wait_until_ready(timeout=120):
        raise RuntimeError("RAG not ready")
    settings.query_rewrite_enabled = False
    q = "宪法规定公民有言论、出版、集会、结社、游行、示威的自由，是哪一条？"
    chunks, _meta = retrieve_fusion(q, profile=False)
    print(f"[live] returned {len(chunks)} chunks (top_k={settings.top_k})")
    for c in chunks:
        print(f"  - {c['article_no']} rerank_score={c.get('rerank_score', c.get('score'))}")
    if chunks and chunks[0].get("rerank_gap_truncate"):
        print(f"  truncate: {chunks[0]['rerank_gap_truncate']}")
    print("[OK] live retrieve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    test_gradual_ladder_drops_tail()
    test_min_k_does_not_bypass_top_relative()
    test_chain_small_steps_cut_by_top_floor()
    test_cliff_after_top_two()
    test_flat_scores_keeps_max_k()
    test_disabled_keeps_all()
    if args.live:
        test_live_retrieve()
    print("\nAll rerank truncate tests passed.")


if __name__ == "__main__":
    main()
