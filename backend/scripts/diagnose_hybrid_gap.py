"""诊断 Cascade 混合检索相对改写单路的差异。

用法:
  cd backend
  python scripts/diagnose_hybrid_gap.py
  python scripts/diagnose_hybrid_gap.py --ids v68,v25
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from config import settings
from query_rewrite import rewrite_for_search
from rag import retrieve_fusion, wait_until_ready
from scripts.compare_rag import (
    load_questions,
    normalize_article_no,
    retrieval_hit,
    run_retrieval,
    target_article_nums,
)


EVAL_FILE = Path(__file__).resolve().parent.parent / "data" / "eval_questions_verified.yaml"


def load_questions_verified() -> list[dict]:
    with open(EVAL_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def rank_in_hits(
    hits: list[dict],
    nums: set[int],
    *,
    law_id: str | None = None,
) -> int | None:
    for i, h in enumerate(hits, start=1):
        if law_id and h.get("law_id") != law_id:
            continue
        n = normalize_article_no(h["article_no"])
        if n in nums:
            return i
    return None


def diagnose_item(item: dict, k: int = 5) -> dict:
    q = item["question"]
    nums = target_article_nums(item)
    law_id = item.get("law_id")
    rw_q, _, _ = rewrite_for_search(q, None)

    _, rw_hit, _ = run_retrieval(rw_q, item, k)
    hybrid_chunks, meta = retrieve_fusion(q, None, top_k=k, rewrite=True)
    hy_hit = retrieval_hit(hybrid_chunks, item)

    if hy_hit and not rw_hit:
        cause = "hybrid_only"
    elif rw_hit and not hy_hit:
        cause = "rewrite_only"
    elif not hy_hit and not rw_hit:
        cause = "both_miss"
    else:
        cause = "both_hit"

    return {
        "id": item["id"],
        "rewrite_hit": rw_hit,
        "hybrid_hit": hy_hit,
        "cause": cause,
        "pool_size": meta.get("rrf_pool_size"),
        "union_size": meta.get("rewrite_union_size"),
        "query_type": meta.get("query_type"),
        "fusion_mode": meta.get("fusion_mode"),
        "rewrite_top": meta.get("rewrite_column_top", []),
        "hybrid_top": [h["article_no"] for h in hybrid_chunks[:k]],
        "expected": item.get("expected_articles"),
        "pool_rank": rank_in_hits(
            [{"law_id": law_id, "article_no": a} for a in meta.get("rewrite_column_top", [])],
            nums,
            law_id=law_id,
        ) if meta.get("rewrite_column_top") else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="")
    args = parser.parse_args()

    print("正在加载 RAG 组件...", flush=True)
    if not wait_until_ready():
        raise SystemExit("RAG 加载超时")

    items = {q["id"]: q for q in load_questions_verified()}
    if args.ids.strip():
        id_list = [x.strip() for x in args.ids.split(",") if x.strip() in items]
    else:
        id_list = sorted(items.keys(), key=lambda x: int(x[1:]))

    rows = []
    for n, qid in enumerate(id_list, start=1):
        print(f"[{n}/{len(id_list)}] {qid} ...", flush=True)
        rows.append(diagnose_item(items[qid]))

    from collections import Counter

    counts = Counter(r["cause"] for r in rows)
    print("\n=== 根因统计 ===")
    for cause, cnt in counts.most_common():
        print(f"  {cause}: {cnt}")

    print("\n=== 混合≠改写 ===")
    for r in rows:
        if r["cause"] in ("hybrid_only", "rewrite_only"):
            print(
                f"{r['id']} {r['cause']} pool={r['pool_size']} union={r['union_size']} "
                f"type={r['query_type']} exp={r['expected']} "
                f"rw={r['rewrite_top']} hy={r['hybrid_top']}"
            )


if __name__ == "__main__":
    main()
