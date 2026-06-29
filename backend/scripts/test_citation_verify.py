"""引用校验单元 / 集成测试（无需完整 68 题评测）。

用法:
  cd backend
  python scripts/test_citation_verify.py           # 纯本地（不调 LLM）
  python scripts/test_citation_verify.py --live    # 含 1 题真实检索+生成+校验（需 API Key）
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
from verify.citations import extract_citations, verify_citations
from verify.repair import apply_fallback, verify_and_repair


def test_extract() -> None:
    text = """**【结论】** 公民有言论自由。

**【解读】** …

**【法律依据】**
- **《中华人民共和国宪法》第三十五条**：…
- **《中华人民共和国刑法》第二百六十四条**：…
"""
    cited = extract_citations(text)
    assert ("中华人民共和国宪法", "第三十五条") in cited
    assert ("中华人民共和国刑法", "第二百六十四条") in cited
    print("[OK] extract_citations")


def test_verify_pass() -> None:
    answer = "**【法律依据】**\n依据 **《中华人民共和国宪法》第三十五条**。"
    chunks = [{"law_name": "中华人民共和国宪法", "article_no": "第三十五条", "text": "…"}]
    result = verify_citations(answer, chunks)
    assert result.passed and not result.hallucination
    print("[OK] verify pass (in kb + in chunks)")


def test_verify_hallucination() -> None:
    answer = "**【法律依据】**\n依据 **《中华人民共和国刑法》第九千九百九十九条**。"
    chunks = [{"law_name": "中华人民共和国宪法", "article_no": "第三十五条", "text": "…"}]
    result = verify_citations(answer, chunks)
    assert not result.passed and result.hallucination
    assert result.invalid[0].reason == "not_in_kb"
    print("[OK] verify detect not_in_kb")


def test_verify_warning_not_in_chunks() -> None:
    answer = "**【法律依据】**\n**《中华人民共和国宪法》第一条**。"
    chunks = [{"law_name": "中华人民共和国宪法", "article_no": "第三十五条", "text": "…"}]
    result = verify_citations(answer, chunks)
    assert result.passed  # 在 KB 中，仅 warning
    assert result.warnings and result.warnings[0].reason == "not_in_retrieved_chunks"
    print("[OK] verify warning not_in_retrieved_chunks")


def test_fallback() -> None:
    bad = "**【结论】** 测试。\n\n**【法律依据】**\n《中华人民共和国刑法》第九千九百九十九条。"
    chunks = [
        {
            "law_name": "中华人民共和国宪法",
            "article_no": "第三十五条",
            "text": "中华人民共和国公民有言论、出版、集会、结社、游行、示威的自由。",
        }
    ]
    fixed = apply_fallback(bad, chunks)
    result = verify_citations(fixed, chunks)
    assert result.passed
    assert "第三十五条" in fixed
    assert "9999" not in fixed
    print("[OK] fallback replaces invalid basis")


def test_verify_and_repair_no_api() -> None:
    settings.citation_verify_repair_enabled = False
    bad = "**【结论】** 测试。\n\n**【法律依据】**\n《中华人民共和国刑法》第九千九百九十九条。"
    chunks = [
        {
            "law_name": "中华人民共和国宪法",
            "article_no": "第三十五条",
            "text": "言论自由",
        }
    ]
    repair = verify_and_repair(bad, chunks, question="言论自由")
    assert repair.action == "fallback_chunks"
    assert repair.citation_verified
    settings.citation_verify_repair_enabled = True
    print("[OK] verify_and_repair fallback (repair disabled)")


def test_live() -> None:
    from rag import answer_question, wait_until_ready

    if not settings.deepseek_api_key:
        print("[SKIP] live test: no DEEPSEEK_API_KEY")
        return
    if not wait_until_ready(timeout=120):
        raise RuntimeError("RAG not ready")
    result = answer_question("宪法规定公民有言论自由是哪一条？")
    assert result.get("is_legal")
    assert "answer" in result
    assert result.get("citation_verified") is True
    print("[OK] live answer_question citation_verified=True")
    print("  preview:", result["answer"][:80].replace("\n", " "), "...")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="含 1 题 API 集成测试")
    args = parser.parse_args()

    test_extract()
    test_verify_pass()
    test_verify_hallucination()
    test_verify_warning_not_in_chunks()
    test_fallback()
    test_verify_and_repair_no_api()
    if args.live:
        test_live()
    print("\nAll citation verify tests passed.")


if __name__ == "__main__":
    main()
