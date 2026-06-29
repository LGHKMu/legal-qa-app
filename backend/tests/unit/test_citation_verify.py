"""引用校验单元测试（纯本地，不调 LLM）。"""

from __future__ import annotations

from config import settings
from verify.citations import extract_citations, verify_citations
from verify.repair import apply_fallback, verify_and_repair


def test_extract_citations() -> None:
    text = """**【结论】** 公民有言论自由。

**【法律依据】**
- **《中华人民共和国宪法》第三十五条**：…
- **《中华人民共和国刑法》第二百六十四条**：…
"""
    cited = extract_citations(text)
    assert ("中华人民共和国宪法", "第三十五条") in cited
    assert ("中华人民共和国刑法", "第二百六十四条") in cited


def test_verify_pass() -> None:
    answer = "**【法律依据】**\n依据 **《中华人民共和国宪法》第三十五条**。"
    chunks = [{"law_name": "中华人民共和国宪法", "article_no": "第三十五条", "text": "…"}]
    result = verify_citations(answer, chunks)
    assert result.passed and not result.hallucination


def test_verify_hallucination() -> None:
    answer = "**【法律依据】**\n依据 **《中华人民共和国刑法》第九千九百九十九条**。"
    chunks = [{"law_name": "中华人民共和国宪法", "article_no": "第三十五条", "text": "…"}]
    result = verify_citations(answer, chunks)
    assert not result.passed and result.hallucination
    assert result.invalid[0].reason == "not_in_kb"


def test_verify_warning_not_in_chunks() -> None:
    answer = "**【法律依据】**\n**《中华人民共和国宪法》第一条**。"
    chunks = [{"law_name": "中华人民共和国宪法", "article_no": "第三十五条", "text": "…"}]
    result = verify_citations(answer, chunks)
    assert result.passed
    assert result.warnings and result.warnings[0].reason == "not_in_retrieved_chunks"


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
