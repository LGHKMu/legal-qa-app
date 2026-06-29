"""引用校验单元测试（纯本地，不调 LLM）。"""

from __future__ import annotations

from config import settings
from verify.citations import extract_citations, select_chunks_cited_in_answer, verify_citations
from verify.repair import apply_fallback, verify_and_repair
from rag import format_citations, sync_citations_for_answer


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


def test_statute_lookup_enforce_extra_citations() -> None:
    """查条场景：【法律依据】中引用了知识库存在但不在 chunks 的法条时应被裁剪。"""
    answer = """**【结论】** 结婚应当男女双方完全自愿。

**【解读】** 本条规范结婚自愿原则。

**【法律依据】**
- **《中华人民共和国民法典》第一千零四十六条**：…
- **《中华人民共和国民法典》第一千零四十七条**：…
"""
    chunks = [
        {
            "law_name": "中华人民共和国民法典",
            "article_no": "第一千零四十六条",
            "text": "结婚应当男女双方完全自愿，禁止任何一方对另一方加以强迫，禁止任何组织或者个人加以干涉。",
        }
    ]
    repair = verify_and_repair(
        answer,
        chunks,
        question="民法典第1046条是什么",
        intent="statute_lookup",
    )
    assert repair.action == "chunks_enforce"
    assert "第一千零四十七条" not in repair.answer
    assert "第一千零四十六条" in repair.answer
    assert repair.citation_verified


def test_statute_lookup_keep_single_citation() -> None:
    answer = "**【法律依据】**\n**《中华人民共和国民法典》第一千零四十六条**：结婚应当男女双方完全自愿。"
    chunks = [
        {
            "law_name": "中华人民共和国民法典",
            "article_no": "第一千零四十六条",
            "text": "结婚应当男女双方完全自愿。",
        }
    ]
    repair = verify_and_repair(
        answer,
        chunks,
        question="民法典第1046条是什么",
        intent="statute_lookup",
    )
    assert repair.action == "pass"
    assert "第一千零四十七条" not in repair.answer


def test_concept_qa_enforce_extra_citations() -> None:
    """概念问答：检索两条但【法律依据】出现额外法条时应裁剪为检索结果。"""
    answer = """**【结论】** 言论自由受宪法保护。

**【解读】** 公民享有言论、出版等自由。

**【法律依据】**
- **《中华人民共和国宪法》第三十五条**：…
- **《中华人民共和国宪法》第三十六条**：…
- **《中华人民共和国宪法》第五十一条**：…
"""
    chunks = [
        {
            "law_name": "中华人民共和国宪法",
            "article_no": "第三十五条",
            "text": "中华人民共和国公民有言论、出版、集会、结社、游行、示威的自由。",
        },
        {
            "law_name": "中华人民共和国宪法",
            "article_no": "第三十六条",
            "text": "中华人民共和国公民有宗教信仰自由。",
        },
    ]
    repair = verify_and_repair(
        answer,
        chunks,
        question="公民有哪些基本权利",
    )
    assert repair.action == "chunks_enforce"
    assert "第五十一条" not in repair.answer
    assert "第三十五条" in repair.answer
    assert "第三十六条" in repair.answer
    assert repair.citation_verified


def test_sync_citations_matches_legal_basis() -> None:
    """检索 5 条但【法律依据】只引用 2 条时，引用卡片应对齐为 2 条。"""
    answer = """**【结论】** 言论与宗教自由受保护。

**【法律依据】**
- **《中华人民共和国宪法》第三十五条**：…
- **《中华人民共和国宪法》第三十六条**：…
"""
    chunks = [
        {"law_name": "中华人民共和国宪法", "article_no": "第三十五条", "text": "言论自由"},
        {"law_name": "中华人民共和国宪法", "article_no": "第三十六条", "text": "宗教自由"},
        {"law_name": "中华人民共和国宪法", "article_no": "第三十七条", "text": "人身自由"},
        {"law_name": "中华人民共和国宪法", "article_no": "第三十八条", "text": "人格尊严"},
        {"law_name": "中华人民共和国宪法", "article_no": "第五十一条", "text": "权利义务"},
    ]
    synced = sync_citations_for_answer(chunks, answer)
    assert len(synced) == 2
    assert {c["article_no"] for c in synced} == {"第三十五条", "第三十六条"}


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
