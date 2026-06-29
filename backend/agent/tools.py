from __future__ import annotations

import json
import time
from functools import lru_cache

from config import DATA_DIR
from context import filter_relevant_history
from query_rewrite import ARTICLE_NO_RE, DOMAIN_TO_LAW_ID

LAW_NAME_PATTERNS: tuple[tuple[str, str], ...] = (
    ("民法典", "civil_code"),
    ("宪法", "constitution"),
    ("刑法", "criminal_law"),
    ("劳动法", "labor_law"),
)


def _parse_article_no(question: str) -> str | None:
    match = ARTICLE_NO_RE.search(question)
    return match.group(0) if match else None


from verify.citations import LAW_ALIASES, article_match


def _parse_law_id(question: str) -> str | None:
    for keyword, law_id in LAW_NAME_PATTERNS:
        if keyword in question:
            return law_id
    for domain, law_id in DOMAIN_TO_LAW_ID.items():
        if domain in question:
            return law_id
    for alias, full_name in LAW_ALIASES.items():
        if alias in question and len(alias) >= 2:
            for keyword, law_id in LAW_NAME_PATTERNS:
                if full_name.endswith(keyword.replace("中华人民共和国", "")) or keyword in full_name:
                    if keyword in question or alias in question:
                        return law_id
    if "《中华人民共和国宪法》" in question or "《宪法》" in question:
        return "constitution"
    if "《中华人民共和国民法典》" in question or "《民法典》" in question:
        return "civil_code"
    if "《中华人民共和国刑法》" in question or "《刑法》" in question:
        return "criminal_law"
    if "《中华人民共和国劳动法》" in question or "《劳动法》" in question:
        return "labor_law"
    return None


@lru_cache(maxsize=8)
def _load_parsed_law(law_id: str) -> dict:
    path = DATA_DIR / "parsed" / f"{law_id}.json"
    if not path.exists():
        return {"articles": []}
    return json.loads(path.read_text(encoding="utf-8"))


def lookup_article(question: str) -> tuple[dict | None, dict]:
    """从 parsed JSON 精确查条；返回 (chunk, summary)。"""
    article_no = _parse_article_no(question)
    if not article_no:
        return None, {"found": False, "reason": "no_article_no"}

    law_id = _parse_law_id(question)
    candidates: list[tuple[str, dict]] = []

    if law_id:
        data = _load_parsed_law(law_id)
        candidates.append((law_id, data))
    else:
        for lid in ("constitution", "civil_code", "criminal_law", "labor_law"):
            candidates.append((lid, _load_parsed_law(lid)))

    for lid, data in candidates:
        law_name = data.get("law_name", "")
        source_url = data.get("source_url", "")
        for art in data.get("articles", []):
            no = art.get("article_no", "")
            if article_match(no, article_no):
                chunk = {
                    "law_id": lid,
                    "law_name": law_name,
                    "article_no": no,
                    "hierarchy": art.get("hierarchy", ""),
                    "text": art.get("text", ""),
                    "source_url": source_url,
                    "score": 1.0,
                }
                return chunk, {
                    "found": True,
                    "law_id": lid,
                    "article_no": no,
                    "source": "parsed_json",
                }

    return None, {"found": False, "article_no": article_no, "law_id": law_id}


def _article_nos(chunks: list[dict]) -> list[str]:
    return [c.get("article_no", "") for c in chunks]


def _search_laws_once(
    question: str,
    history: list[dict],
    state,
    trace,
    *,
    rewrite: bool | None = None,
    search_query_override: str | None = None,
) -> tuple[list[dict], dict]:
    from rag import retrieve_fusion

    return retrieve_fusion(
        question,
        history,
        profile=trace is not None,
        rewrite=rewrite,
        search_query_override=search_query_override,
    )


def _apply_search_results(state, chunks: list[dict], meta: dict) -> None:
    from rag import format_citations

    state.chunks = chunks
    state.citations = format_citations(chunks)
    state.retrieve_meta = meta


def _inject_topic_anchor_chunks(question: str, chunks: list[dict]) -> list[dict]:
    """命中高频主题时，将核心法条插入结果（避免 top_k 截断丢失关键条）。"""
    from config import settings
    from query_rewrite import topic_anchor_lookup_questions

    lookup_questions = topic_anchor_lookup_questions(question)
    if not lookup_questions:
        return chunks

    seen = {
        (c.get("doc_id") or ("article", c.get("law_name", ""), c.get("article_no", "")))
        for c in chunks
    }
    existing_nos = {c.get("article_no") for c in chunks if c.get("article_no")}
    anchors: list[dict] = []
    for lookup_q in lookup_questions:
        chunk, _ = lookup_article(lookup_q)
        if not chunk:
            continue
        article_no = chunk.get("article_no")
        if article_no and article_no in existing_nos:
            continue
        key = chunk.get("doc_id") or ("article", chunk.get("law_name", ""), article_no)
        if key in seen:
            continue
        anchors.append({**chunk, "topic_anchor": True, "score": max(chunk.get("score", 0.9), 0.9)})
        seen.add(key)
        if article_no:
            existing_nos.add(article_no)

    if not anchors:
        return chunks

    k = settings.top_k
    merged = anchors + chunks
    if len(merged) <= k:
        return merged
    merged.sort(
        key=lambda c: (
            1 if c.get("topic_anchor") else 0,
            c.get("rerank_score") or c.get("score") or -1.0,
        ),
        reverse=True,
    )
    return merged[:k]


def _search_laws_with_case_retry(
    question: str,
    history: list[dict],
    state,
    trace,
) -> dict:
    from agent.retrieval_quality import assess_retrieval_quality, merge_retrieval_chunks
    from rag import build_retrieve_trace_output
    from config import settings

    primary_chunks, primary_meta = _search_laws_once(
        question, state.relevant_history or history, state, trace, rewrite=True
    )
    primary_chunks = _inject_topic_anchor_chunks(question, primary_chunks)
    quality = assess_retrieval_quality(primary_chunks, primary_meta, question=question)

    state.retrieval_attempts = 1
    state.retrieval_retry = False
    state.retrieval_retry_reason = ""
    state.retrieval_retry_strategy = ""

    chunks = primary_chunks
    meta = primary_meta

    if (
        settings.agent_case_retry_enabled
        and not quality.sufficient
    ):
        from query_rewrite import build_case_retry_query

        retry_query = build_case_retry_query(question, primary_meta)
        secondary_chunks, secondary_meta = _search_laws_once(
            question,
            state.relevant_history or history,
            state,
            trace,
            rewrite=False,
            search_query_override=retry_query,
        )
        chunks = merge_retrieval_chunks(primary_chunks, secondary_chunks)
        chunks = _inject_topic_anchor_chunks(question, chunks)
        meta = {
            **primary_meta,
            "retry": True,
            "retry_reason": quality.reason,
            "retry_strategy": "baseline_no_rewrite",
            "retry_query": retry_query,
            "retry_quality_signals": quality.signals,
            "primary_articles": _article_nos(primary_chunks),
            "secondary_articles": _article_nos(secondary_chunks),
            "secondary_query_source": secondary_meta.get("query_source"),
        }
        state.retrieval_attempts = 2
        state.retrieval_retry = True
        state.retrieval_retry_reason = quality.reason
        state.retrieval_retry_strategy = "baseline_no_rewrite"

        if trace:
            trace.step(
                "agent:search_retry",
                0,
                {
                    "reason": quality.reason,
                    "strategy": "baseline_no_rewrite",
                    "signals": quality.signals,
                    "primary_top": meta["primary_articles"][:5],
                    "secondary_top": meta["secondary_articles"][:5],
                    "merged_count": len(chunks),
                },
            )

    _apply_search_results(state, chunks, meta)
    out = build_retrieve_trace_output(state.citations, meta, chunks)
    out["attempts"] = state.retrieval_attempts
    out["retry"] = state.retrieval_retry
    if state.retrieval_retry:
        out["retry_reason"] = state.retrieval_retry_reason
        out["retry_strategy"] = state.retrieval_retry_strategy
        out["quality_signals"] = quality.signals
    return out


def run_tool(
    tool: str,
    *,
    question: str,
    history: list[dict],
    state,
    trace=None,
    intent: str = "concept_qa",
) -> dict:
    t0 = time.perf_counter()

    if tool == "filter_context":
        relevant = filter_relevant_history(question, history)
        state.relevant_history = relevant
        out = {"context_turns": len(relevant)}
        if trace:
            trace.step("context_filter", (time.perf_counter() - t0) * 1000, out)
        return out

    if tool == "get_article":
        chunk, summary = lookup_article(question)
        if chunk:
            state.chunks = [chunk]
            from rag import format_citations

            state.citations = format_citations(state.chunks)
        out = {**summary, "citation_count": len(state.citations)}
        if trace:
            trace.step("agent:get_article", (time.perf_counter() - t0) * 1000, out)
        return out

    if tool == "search_laws":
        from rag import build_retrieve_trace_output

        if intent == "case_consult":
            out = _search_laws_with_case_retry(
                question, history, state, trace
            )
        else:
            chunks, meta = _search_laws_once(
                question,
                state.relevant_history or history,
                state,
                trace,
            )
            _apply_search_results(state, chunks, meta)
            out = build_retrieve_trace_output(state.citations, meta, chunks)

        if trace:
            trace.step("retrieve", (time.perf_counter() - t0) * 1000, out)
        return out

    if tool == "generate_answer":
        state.is_legal = intent != "non_legal"
        out = {"is_legal": state.is_legal, "chunk_count": len(state.chunks)}
        if trace:
            trace.step("classify", (time.perf_counter() - t0) * 1000, {"is_legal": state.is_legal})
        return out

    if tool == "verify_citations":
        from verify.repair import verify_and_repair

        repair = verify_and_repair(
            state.answer_text,
            state.chunks,
            question=question,
            history=state.relevant_history,
            trace=trace,
            intent=intent,
        )
        state.repair = repair
        state.answer_text = repair.answer
        state.citation_verified = repair.citation_verified
        from rag import sync_citations_for_answer

        state.citations = sync_citations_for_answer(state.chunks, state.answer_text)
        out = repair.verify.to_trace_output()
        out["action"] = repair.action
        out["citation_verified"] = repair.citation_verified
        out["citation_count"] = len(state.citations)
        if trace:
            trace.step("agent:verify", (time.perf_counter() - t0) * 1000, out)
        return out

    return {}
