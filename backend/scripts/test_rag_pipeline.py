"""RAG 检索链路测试脚本（方案五 + 方案七）。

用法:
  cd backend

  # 1. 冒烟测试（不调 LLM，约 5 秒）
  python scripts/test_rag_pipeline.py --smoke

  # 2. 单题详细诊断（需 DEEPSEEK_API_KEY）
  python scripts/test_rag_pipeline.py --id v05

  # 3. 多题对比 baseline / 改写 / Cascade 混合
  python scripts/test_rag_pipeline.py --ids v05,v26,v41

  # 4. 口语化问法子集（eval_questions_verified 中 oral_verified）
  python scripts/test_rag_pipeline.py --oral
  # 5. 保存 JSON
  python scripts/test_rag_pipeline.py --oral --output data/test_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from query_rewrite import clear_rewrite_cache, rewrite_for_search, rewrite_query_two_stage
from rag import (
    build_retrieval_query,
    build_search_query,
    retrieve,
    retrieve_fusion,
    wait_until_ready,
)
from retrieval.fusion import rrf_select_topk
from scripts.compare_rag import (
    hybrid_column_label,
    load_questions,
    print_retrieval_config,
    retrieval_hit,
)


def _hit_mark(ok: bool) -> str:
    return "命中" if ok else "未命中"


def run_smoke() -> None:
    """纯本地逻辑，不加载 LLM / 不调 API。"""
    print("=== 冒烟测试（RRF 融合逻辑）===\n")

    def hit(doc_id: str, ranked: list[str]) -> bool:
        return doc_id in ranked

    # 模拟 q05：改写路排名第 5 的 doc 应被保留
    rewrite_hits = [{"doc_id": f"rw_{i}", "article_no": f"R{i}"} for i in range(5)]
    rewrite_hits[4]["doc_id"] = "target_43"
    rewrite_hits[4]["article_no"] = "第四十三条"
    base_hits = [{"doc_id": f"base_{i}", "article_no": f"B{i}"} for i in range(5)]

    fused = rrf_select_topk(base_hits, rewrite_hits, top_k=5, rrf_k=60)
    fused_ids = [d for d, _ in fused]
    ok = "target_43" in fused_ids
    print(f"改写路末位 doc 是否保留: {'通过' if ok else '失败'}")
    print(f"融合顺序: {fused_ids}\n")

    # 要素拼接
    from query_rewrite import LegalElements, build_query_from_elements

    elements = LegalElements(
        domains=["宪法", "劳动法"],
        query_keywords=["言论自由", "停职", "劳动者权利"],
        topics=["停职"],
    )
    q = build_query_from_elements(elements)
    print(f"两阶段 query 拼接: {q}")
    print("冒烟测试完成。\n")


def diagnose_question(item: dict) -> dict:
    """单题诊断，返回结果 dict。"""
    qid = item["id"]
    question = item["question"]
    expected = item["expected_articles"]

    result: dict = {
        "id": qid,
        "question": question,
        "expected_articles": expected,
    }

    base_q = build_retrieval_query(question)
    base_chunks = retrieve(base_q, top_k=5)
    base_arts = [c["article_no"] for c in base_chunks]
    result["baseline"] = {
        "query": base_q,
        "articles": base_arts,
        "hit": retrieval_hit(base_chunks, item),
    }

    rw_q, src, elements = rewrite_for_search(question)
    rw_chunks = retrieve(rw_q or base_q, top_k=5)
    result["rewrite"] = {
        "query": rw_q,
        "source": src,
        "articles": [c["article_no"] for c in rw_chunks],
        "hit": retrieval_hit(rw_chunks, item),
        "legal_elements": elements.to_dict() if elements else None,
    }

    chunks, meta = retrieve_fusion(question, rewrite=True)
    fusion_arts = [c["article_no"] for c in chunks]
    result["hybrid"] = {
        "fusion_mode": meta.get("fusion_mode"),
        "rrf_pool_size": meta.get("rrf_pool_size"),
        "rewrite_union_size": meta.get("rewrite_union_size"),
        "query_type": meta.get("query_type"),
        "rewrite_query": meta.get("rewrite_query"),
        "articles": fusion_arts,
        "hit": retrieval_hit(chunks, item),
    }
    return result


def print_result(r: dict) -> None:
    print(f"\n[{r['id']}] {r['question']}")
    print(f"  期望: {r['expected_articles']}")
    b = r["baseline"]
    print(f"  baseline ({_hit_mark(b['hit'])}): {b['query'][:50]}")
    print(f"    -> {b['articles']}")

    rw = r["rewrite"]
    print(f"  rewrite ({_hit_mark(rw['hit'])}): {rw['query'][:50] if rw['query'] else '-'}")
    print(f"    source={rw['source']}  -> {rw['articles']}")
    if rw.get("legal_elements"):
        print(f"    elements={rw['legal_elements']}")

    f = r["hybrid"]
    pool = f.get("rrf_pool_size")
    union = f.get("rewrite_union_size")
    extra = f", pool={pool}" if pool else ""
    if union:
        extra += f", union={union}"
    print(
        f"  hybrid[{f.get('fusion_mode', 'cascade')}] ({_hit_mark(f['hit'])}): "
        f"type={f.get('query_type')}{extra}"
    )
    if f.get("rewrite_query"):
        print(f"    rewrite_q={f['rewrite_query'][:50]}")
    print(f"    -> {f['articles']}")
    if rw.get("hit") and not f["hit"]:
        print("    ⚠ 改写命中但 Cascade 混合未命中")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索链路测试")
    parser.add_argument("--smoke", action="store_true", help="本地冒烟，不调 LLM")
    parser.add_argument("--id", type=str, default="", help="单题 id，如 v05")
    parser.add_argument("--ids", type=str, default="", help="多题逗号分隔，如 v05,v26,v41")
    parser.add_argument("--oral", action="store_true", help="口语化问法子集（oral_verified）")
    parser.add_argument("--output", type=str, default="", help="保存 JSON 路径")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return

    print("正在加载 Embedding 模型...", flush=True)
    if not wait_until_ready():
        raise RuntimeError("RAG 组件加载超时")
    if not settings.deepseek_api_key:
        raise RuntimeError("需要配置 backend/.env 中的 DEEPSEEK_API_KEY")

    questions = load_questions()
    if args.id:
        selected = [q for q in questions if q["id"] == args.id]
        if not selected:
            raise SystemExit(f"未找到题目: {args.id}")
    elif args.ids:
        ids = {x.strip() for x in args.ids.split(",") if x.strip()}
        selected = [q for q in questions if q["id"] in ids]
        if not selected:
            raise SystemExit(f"未找到题目: {', '.join(sorted(ids))}")
    elif args.oral:
        selected = [q for q in questions if q.get("group") == "oral_verified"]
        if not selected:
            raise SystemExit("未找到 oral_verified 题目")
    else:
        selected = questions[:5]
        print("未指定题目，默认测试前 5 题。可用 --id v05 或 --oral\n")

    print(f"测试 {len(selected)} 题，rewrite_mode={settings.query_rewrite_mode}")
    print_retrieval_config()
    print(flush=True)

    results = []
    summary = {"baseline": 0, "rewrite": 0, "hybrid": 0, "n": len(selected)}

    for item in selected:
        clear_rewrite_cache()
        r = diagnose_question(item)
        results.append(r)
        print_result(r)
        if r["baseline"]["hit"]:
            summary["baseline"] += 1
        if r["rewrite"]["hit"]:
            summary["rewrite"] += 1
        if r["hybrid"]["hit"]:
            summary["hybrid"] += 1

    n = summary["n"]
    hybrid_label = hybrid_column_label()
    print("\n" + "=" * 48)
    print(f"汇总 (N={n})  [{hybrid_label}]")
    print(f"  baseline Recall@5: {summary['baseline']/n:.1%} ({summary['baseline']}/{n})")
    print(f"  rewrite   Recall@5: {summary['rewrite']/n:.1%} ({summary['rewrite']}/{n})")
    print(f"  hybrid    Recall@5: {summary['hybrid']/n:.1%} ({summary['hybrid']}/{n})")
    print("=" * 48)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query_rewrite_mode": settings.query_rewrite_mode,
            "retrieval": "cascade_union",
            "summary": summary,
            "results": results,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
