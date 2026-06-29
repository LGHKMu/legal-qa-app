"""引用校验失败后的修复：LLM 重写一次 → 检索法条兜底。"""
from __future__ import annotations

import time
from dataclasses import dataclass

from config import settings
from verify.citations import LEGAL_BASIS_MARK, VerifyResult, verify_citations

FALLBACK_NOTE = (
    "\n\n*以下【法律依据】已替换为本次检索到的法条原文摘要；"
    "原回答中未能核实的引用已移除。*"
)


@dataclass
class RepairResult:
    answer: str
    citation_verified: bool
    verify: VerifyResult
    action: str  # pass | rewrite_once | fallback_chunks | disabled


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


def verify_and_repair(
    answer: str,
    chunks: list[dict],
    *,
    question: str = "",
    history: list[dict] | None = None,
    trace=None,
) -> RepairResult:
    """校验引用；失败则尝试 LLM 修正【法律依据】，仍失败则用检索 chunks 兜底。"""
    t0 = time.perf_counter()
    if not settings.citation_verify_enabled:
        verify = verify_citations(answer, chunks)
        result = RepairResult(
            answer=answer,
            citation_verified=verify.passed,
            verify=verify,
            action="disabled",
        )
        _trace_verify(trace, t0, result)
        return result

    verify = verify_citations(answer, chunks)
    if verify.passed:
        result = RepairResult(
            answer=answer,
            citation_verified=True,
            verify=verify,
            action="pass",
        )
        _trace_verify(trace, t0, result)
        return result

    current = answer
    action = "pass"

    if settings.citation_verify_repair_enabled and question:
        from llm import repair_legal_citations

        invalid_desc = "；".join(
            f"《{i.law}》{i.article_no}({i.reason})" for i in verify.invalid
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
                action = "rewrite_once"
                result = RepairResult(
                    answer=current,
                    citation_verified=True,
                    verify=verify,
                    action=action,
                )
                _trace_verify(trace, t0, result)
                return result
            action = "rewrite_once"

    current = apply_fallback(current, chunks)
    verify = verify_citations(current, chunks)
    result = RepairResult(
        answer=current,
        citation_verified=verify.passed,
        verify=verify,
        action="fallback_chunks",
    )
    _trace_verify(trace, t0, result)
    return result


def _trace_verify(trace, t0: float, result: RepairResult) -> None:
    if trace is None:
        return
    output = result.verify.to_trace_output()
    output["action"] = result.action
    output["citation_verified"] = result.citation_verified
    trace.step("verify", (time.perf_counter() - t0) * 1000, output)
