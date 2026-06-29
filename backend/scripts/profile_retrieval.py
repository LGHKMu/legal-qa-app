"""Cascade 检索分阶段耗时剖析（需 DEEPSEEK_API_KEY 做改写）。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from config import settings
from rag import (
    enable_retrieval_profile,
    get_retrieval_profile,
    retrieve_fusion,
    wait_until_ready,
    warmup,
)

PROFILE_KEYS = [
    "rewrite_api_ms",
    "embed_ms",
    "chroma_ms",
    "dual_bm25_ms",
    "pool_build_ms",
    "hybrid_rerank_ms",
    "rewrite_col_ms",
    "rewrite_col_bm25_ms",
    "rewrite_col_rerank_ms",
    "union_rerank_ms",
]


def load_questions(path: Path, limit: int) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("questions", [])
    return items[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascade 检索分阶段耗时")
    parser.add_argument(
        "--eval",
        type=str,
        default="data/eval_questions_verified.yaml",
        help="评测集路径",
    )
    parser.add_argument("-n", "--limit", type=int, default=10, help="抽样题数")
    args = parser.parse_args()

    warmup()
    if not wait_until_ready(120):
        raise SystemExit("RAG 组件加载超时")

    eval_path = Path(__file__).resolve().parent.parent / args.eval
    questions = load_questions(eval_path, args.limit)

    totals: dict[str, float] = {k: 0.0 for k in PROFILE_KEYS}
    wall_ms: list[float] = []
    union_skipped = 0
    union_rerank = 0

    print(f"配置: pool_k={settings.rrf_pool_k}, rerank={settings.rerank_enabled}")
    print(f"剖析 {len(questions)} 题...\n")

    for item in questions:
        q = item["question"]
        enable_retrieval_profile(True)
        t0 = time.perf_counter()
        _, meta = retrieve_fusion(q, None, top_k=5, rewrite=True)
        wall = (time.perf_counter() - t0) * 1000
        wall_ms.append(wall)
        prof = get_retrieval_profile()
        for k in PROFILE_KEYS:
            totals[k] += prof.get(k, 0.0)
        if meta.get("union_rerank_skipped"):
            union_skipped += 1
        if meta.get("union_rerank"):
            union_rerank += 1

    n = len(questions)
    avg_wall = sum(wall_ms) / n
    print(f"平均墙钟时延: {avg_wall:.0f} ms\n")
    print("| 阶段 | 平均 ms | 占墙钟 |")
    print("|------|---------|--------|")

    labeled = {
        "rewrite_api_ms": "① Query 改写 (DeepSeek)",
        "embed_ms": "② Embedding（含在 dual 内）",
        "chroma_ms": "③ Chroma 向量检索",
        "dual_bm25_ms": "④ 多路 BM25",
        "pool_build_ms": "⑤ Cascade 建池",
        "hybrid_rerank_ms": "⑥ 混合池精排 (Cross-Encoder)",
        "rewrite_col_ms": "⑦ 改写列（BM25+精排）",
        "rewrite_col_rerank_ms": "  - rewrite col rerank",
        "union_rerank_ms": "⑧ Union 并集精排",
    }
    profile_sum = sum(totals[k] for k in labeled)
    for key, label in labeled.items():
        avg = totals[key] / n
        pct = avg / avg_wall * 100 if avg_wall else 0
        print(f"| {label} | {avg:.0f} | {pct:.0f}% |")

    other = max(0.0, avg_wall - profile_sum / n)
    print(f"| （未计入/重叠/开销） | {other:.0f} | {other/avg_wall*100:.0f}% |")
    print()
    print(f"Union 跳过并集精排: {union_skipped}/{n} 题")
    print(f"Union 执行并集精排: {union_rerank}/{n} 题")


if __name__ == "__main__":
    main()
