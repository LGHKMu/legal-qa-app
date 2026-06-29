"""校验评测集标注是否落在知识库内，并输出报告。

用法:
  cd backend
  python scripts/verify_eval_dataset.py
  python scripts/verify_eval_dataset.py --file data/eval_questions_verified.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import get_collection, wait_until_ready
from scripts.compare_rag import normalize_article_no


def load_kb_index() -> dict[tuple[str, int], str]:
    collection = get_collection()
    data = collection.get(include=["metadatas"])
    index: dict[tuple[str, int], str] = {}
    for meta in data["metadatas"]:
        num = normalize_article_no(meta["article_no"])
        if num is not None:
            index[(meta["law_id"], num)] = meta["article_no"]
    return index


def verify_file(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    questions = payload.get("questions", [])
    kb = load_kb_index()

    errors: list[str] = []
    warnings: list[str] = []

    for q in questions:
        qid = q.get("id", "?")
        law_id = q.get("law_id")
        if not law_id:
            errors.append(f"{qid}: 缺少 law_id")
            continue
        if not q.get("source_url"):
            warnings.append(f"{qid}: 缺少 source_url（建议补充出处）")
        for exp in q.get("expected_articles", []):
            num = normalize_article_no(exp)
            if num is None:
                errors.append(f"{qid}: 无法解析条号 {exp!r}")
                continue
            if (law_id, num) not in kb:
                errors.append(f"{qid}: 知识库无 {law_id} 第{num}条（标注 {exp!r}）")
        for alt in q.get("acceptable_articles", []) or []:
            num = normalize_article_no(alt)
            if num is None:
                errors.append(f"{qid}: 无法解析备选条号 {alt!r}")
            elif (law_id, num) not in kb:
                errors.append(f"{qid}: 知识库无备选 {law_id} 第{num}条")

    print(f"文件: {path}")
    print(f"题数: {len(questions)}")
    print(f"知识库索引: {len(kb)} 条 (law_id, 条号)")
    if warnings:
        print(f"\n警告 ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print(f"\n错误 ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\n校验通过：全部期望法条均存在于知识库。")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default=str(Path(__file__).resolve().parent.parent / "data" / "eval_questions_verified.yaml"),
    )
    args = parser.parse_args()
    if not wait_until_ready():
        raise SystemExit("RAG 组件加载超时")
    raise SystemExit(verify_file(Path(args.file)))


if __name__ == "__main__":
    main()
