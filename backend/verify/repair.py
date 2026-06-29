"""引用校验失败后的修复：LLM 重写一次 → 检索法条兜底。"""
from __future__ import annotations

import time
from dataclasses import dataclass

from config import settings
from verify.citations import LEGAL_BASIS_MARK, VerifyResult, select_chunks_cited_in_answer, verify_citations

FALLBACK_NOTE = (
    "\n\n*以下【法律依据】已替换为本次检索到的法条原文摘要；"
    "原回答中未能核实的引用已移除。*"
)


@dataclass
class RepairResult:
    answer: str
    citation_verified: bool
    verify: VerifyResult
    action: str  # pass | rewrite_once | fallback_chunks | chunks_enforce | disabled


def _split_answer_sections(answer: str) -> tuple[str, str]:
    """返回 (法律依据之前的内容, 法律依据段或空)。"""
    if LEGAL_BASIS_MARK not in answer:
        return answer, ""
    before, basis = answer.split(LEGAL_BASIS_MARK, 1)
    return before.rstrip(), LEGAL_BASIS_MARK + basis.lstrip()


def build_fallback_legal_basis(chunks: list[dict]) -> str:
    lines = [LEGAL_BASIS_MARK]
    for chunk in chunks:
        text = chunk.get("text", "").replace("\n", " ")
        snippet = text[:120] + ("…" if len(text) > 120 else "")
        lines.append(
            f"- **《{chunk['law_name']}》{chunk['article_no']}**：{snippet}"
        )
    return "\n".join(lines)


def apply_fallback(answer: str, chunks: list[dict]) -> str:
    before, _ = _split_answer_sections(answer)
    basis = build_fallback_legal_basis(chunks)
    if before:
        return f"{before}\n\n{basis}{FALLBACK_NOTE}"
    return f"{basis}{FALLBACK_NOTE}"


def enforce_chunks_legal_basis(answer: str, chunks: list[dict]) -> str:
    """【法律依据】仅保留回答中引用且属于本次检索 chunks 的法条。"""
    before, _ = _split_answer_sections(answer)
    matched = select_chunks_cited_in_answer(chunks, answer)
    if not matched:
        matched = chunks
    basis = build_fallback_legal_basis(matched)
    if before:
        return f"{before}\n\n{basis}"
    return basis


def _enforce_if_extra_citations(
    answer: str,
    chunks: list[dict],
    verify: VerifyResult,
    action: str,
) -> RepairResult:
    if chunks and verify.warnings:
        enforced = enforce_chunks_legal_basis(answer, chunks)
        verify = verify_citations(enforced, chunks)
        return RepairResult(
            answer=enforced,
            citation_verified=verify.passed,
            verify=verify,
            action="chunks_enforce",
        )
    return RepairResult(
        answer=answer,
        citation_verified=verify.passed,
        verify=verify,
        action=action,
    )


def verify_and_repair(
    answer: str,
    chunks: list[dict],
    *,
    question: str = "",
    history: list[dict] | None = None,
    trace=None,
    intent: str = "",
) -> RepairResult:
    """校验引用；失败则尝试 LLM 修正【法律依据】，仍失败则用检索 chunks 兜底。"""
    del intent  # 保留参数以兼容调用方，逻辑已统一按 chunks 裁剪
    t0 = time.perf_counter()
    if not settings.citation_verify_enabled:
        verify = verify_citations(answer, chunks)
        result = _enforce_if_extra_citations(answer, chunks, verify, "disabled")
        _trace_verify(trace, t0, result)
        return result

    verify = verify_citations(answer, chunks)
    if verify.passed:
        result = _enforce_if_extra_citations(answer, chunks, verify, "pass")
        _trace_verify(trace, t0, result)
        return result

    current = answer

    if verify.warnings and not verify.invalid and chunks:
        result = _enforce_if_extra_citations(current, chunks, verify, "chunks_enforce")
        _trace_verify(trace, t0, result)
        return result

    if settings.citation_verify_repair_enabled and question:
        from llm import repair_legal_citations

        issues = verify.invalid + verify.warnings
        invalid_desc = "；".join(
            f"《{i.law}》{i.article_no}({i.reason})" for i in issues
        )
        repaired = repair_legal_citations(
            question,
            current,
            chunks,
            invalid_desc,
            history,
        )
        if repaired and repaired.strip() != current.strip():
            current = repaired
            verify = verify_citations(current, chunks)
            if verify.passed:
                result = _enforce_if_extra_citations(
                    current, chunks, verify, "rewrite_once"
                )
                _trace_verify(trace, t0, result)
                return result

    current = apply_fallback(current, chunks)
    verify = verify_citations(current, chunks)
    result = _enforce_if_extra_citations(current, chunks, verify, "fallback_chunks")
    _trace_verify(trace, t0, result)
    return result


def _trace_verify(trace, t0: float, result: RepairResult) -> None:
    if trace is None:
        return
    output = result.verify.to_trace_output()
    output["action"] = result.action
    output["citation_verified"] = result.citation_verified
    trace.step("verify", (time.perf_counter() - t0) * 1000, output)
