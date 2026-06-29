from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import build_index


def main() -> None:
    counts = build_index()
    print("向量库构建完成：")
    for law_id, count in counts.items():
        print(f"  - {law_id}: {count} 条")
    if counts:
        print("BM25 索引已同步构建（data/bm25/）")


if __name__ == "__main__":
    main()
