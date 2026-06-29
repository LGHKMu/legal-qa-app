"""精排截断单元测试（相对首条 α + 相邻跌幅 γ）。"""

from __future__ import annotations

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
    settings.rerank_gap_truncate_enabled = True
    settings.rerank_truncate_min_relative = 0.78
    settings.rerank_truncate_max_step_drop = 0.22
    hits = [_hit("a", 0.98), _hit("b", 0.76), _hit("c", 0.55), _hit("d", 0.43)]
    kept = truncate_by_score_gap(hits, max_k=5, min_k=1)
    assert len(kept) == 1
    assert kept[0]["article_no"] == "a"


def test_min_k_does_not_bypass_top_relative() -> None:
    settings.rerank_gap_truncate_enabled = True
    hits = [_hit("a", 0.98), _hit("b", 0.60), _hit("c", 0.58)]
    kept = truncate_by_score_gap(
        hits, max_k=5, min_k=2, min_relative=0.72, max_step_drop=0.25
    )
    assert len(kept) == 1
    assert kept[0]["article_no"] == "a"


def test_chain_small_steps_cut_by_top_floor() -> None:
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


def test_flat_scores_keeps_max_k() -> None:
    hits = [_hit(f"第{i}条", 0.90 - i * 0.02) for i in range(1, 6)]
    kept = truncate_by_score_gap(hits, max_k=5, min_k=1)
    assert len(kept) == 5


def test_disabled_keeps_all() -> None:
    settings.rerank_gap_truncate_enabled = False
    hits = [_hit("第三十五条", 0.98), _hit("第四十七条", 0.43)]
    kept = truncate_by_score_gap(hits, max_k=5)
    assert len(kept) == 2
    settings.rerank_gap_truncate_enabled = True
