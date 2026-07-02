"""RAG vs 无 RAG 对比评测脚本。

用法:
  cd backend
  python scripts/compare_rag.py                  # 完整评测（需 DEEPSEEK_API_KEY）
  python scripts/compare_rag.py --retrieval-only # 仅检索指标
  python scripts/compare_rag.py --compare-rewrite --retrieval-only  # baseline / 改写 / 混合 / Agent线上
  python scripts/compare_rag.py --compare-agent --retrieval-only   # 仅 Agent 线上路径
  python scripts/compare_rag.py --no-rewrite     # 关闭 Query 改写
  python scripts/compare_rag.py --output data/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 避免 Windows 终端加载模型时进度条假死
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BACKEND = Path(__file__).resolve().parent.parent

from config import settings
from agent.pre_retrieval import run_agent_pre_retrieval
from llm import ask_llm, ask_llm_no_rag
from rag import (
    build_retrieval_query,
    build_search_query,
    get_collection,
    retrieve,
    retrieve_fusion,
    wait_until_ready,
)

EVAL_FILE = Path(__file__).resolve().parent.parent / "data" / "eval_questions_verified.yaml"

GATE_MODE_ALIASES: dict[str, str] = {
    "retrieval": "rag_retrieval",
    "agent": "retrieval_agent",
    "hybrid": "retrieval_hybrid",
    "baseline": "retrieval_baseline",
    "rewrite": "retrieval_rewrite",
}
CITATION_RE = re.compile(r"《([^》]{2,30})》\s*(第[零〇一二三四五六七八九十百千万\d]+条)")
CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def cn_to_int(text: str) -> int:
    if text.isdigit():
        return int(text)
    total = 0
    section = 0
    number = 0
    for char in text:
        if char in CN_DIGITS:
            number = CN_DIGITS[char]
        elif char == "十":
            section += (number or 1) * 10
            number = 0
        elif char == "百":
            section += (number or 1) * 100
            number = 0
        elif char == "千":
            section += (number or 1) * 1000
            number = 0
        elif char == "万":
            total += (section + number) * 10000
            section = 0
            number = 0
    return total + section + number


def normalize_article_no(article_no: str) -> int | None:
    match = re.fullmatch(r"第(.+?)条", article_no.strip())
    if not match:
        return None
    return cn_to_int(match.group(1))


def article_match(a: str, b: str) -> bool:
    na, nb = normalize_article_no(a), normalize_article_no(b)
    return na is not None and na == nb


# 宽松命中：除 expected / acceptable 外，同 law_id 下允许期望条号 ±N（默认 2）
EVAL_HIT_TOLERANCE = 2


def target_article_nums(item: dict, *, tolerance: int = EVAL_HIT_TOLERANCE) -> set[int]:
    """评测认可的条号集合：主标 + 备选 + 主标条号 ±tolerance。"""
    nums: set[int] = set()
    expected_raw = item.get("expected_articles", [])
    for art in expected_raw + list(item.get("acceptable_articles") or []):
        n = normalize_article_no(art)
        if n is not None:
            nums.add(n)
    for art in expected_raw:
        n = normalize_article_no(art)
        if n is None:
            continue
        for delta in range(1, tolerance + 1):
            if n - delta > 0:
                nums.add(n - delta)
            nums.add(n + delta)
    return nums


def retrieval_hit(
    chunks: list[dict],
    item: dict,
    *,
    tolerance: int = EVAL_HIT_TOLERANCE,
) -> bool:
    """Recall@K 命中：law_id 一致，且条号落在宽松认可集合内。"""
    law_id = item.get("law_id")
    nums = target_article_nums(item, tolerance=tolerance)
    if not nums:
        return False
    for c in chunks:
        if law_id and c.get("law_id") != law_id:
            continue
        n = normalize_article_no(c.get("article_no", ""))
        if n is not None and n in nums:
            return True
    return False


def any_expected_hit(retrieved: list[str], expected: list[str]) -> bool:
    """仅条号匹配（无 law_id）；保留给无 chunk 元数据的旧调用。"""
    return any(article_match(r, e) for r in retrieved for e in expected)


@dataclass
class QuestionResult:
    id: str
    question: str
    mode: str
    recall_at_k: bool | None = None
    retrieved_articles: list[str] = field(default_factory=list)
    cited_articles: list[str] = field(default_factory=list)
    citation_recall: float = 0.0
    citation_precision: float = 0.0
    hallucination: bool = False
    latency_ms: float = 0.0
    answer_preview: str = ""
    search_query: str = ""
    query_source: str = ""
    fusion_mode: str = ""
    rrf_pool_size: int = 0
    intent: str = ""
    route_source: str = ""
    retrieval_retry: bool = False
    retrieval_retry_reason: str = ""
    tools_run: list[str] = field(default_factory=list)


@dataclass
class Summary:
    mode: str
    count: int
    recall_at_k: float | None = None
    avg_citation_recall: float = 0.0
    avg_citation_precision: float = 0.0
    hallucination_rate: float = 0.0
    avg_latency_ms: float = 0.0


def load_questions(eval_file: Path | None = None) -> list[dict]:
    path = eval_file or EVAL_FILE
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def normalize_gate_mode(mode: str) -> str:
    key = mode.strip()
    return GATE_MODE_ALIASES.get(key, key)


def recall_for_gate_mode(summaries: list[Summary], gate_mode: str) -> float | None:
    """从 summaries 中取指定模式的 Recall@K。"""
    target = normalize_gate_mode(gate_mode)
    for summary in summaries:
        if summary.mode == target:
            return summary.recall_at_k
    return None


def enforce_recall_gate(
    summaries: list[Summary],
    *,
    gate_mode: str,
    min_recall: float,
) -> None:
    """低于阈值时抛出 SystemExit(1)。"""
    actual = recall_for_gate_mode(summaries, gate_mode)
    target = normalize_gate_mode(gate_mode)
    if actual is None:
        print(f"\n[FAIL] 门禁模式 {target} 无评测汇总", flush=True)
        raise SystemExit(1)
    print(
        f"\n[Gate] {target} Recall@{settings.top_k}: {actual:.1%} "
        f"(阈值 >= {min_recall:.1%})",
        flush=True,
    )
    if actual + 1e-9 < min_recall:
        print(f"[FAIL] Recall 未达门禁阈值", flush=True)
        raise SystemExit(1)
    print("[PASS] Recall 门禁通过", flush=True)



def agent_column_label() -> str:
    """第四列：与 /api/ask Agent 前置检索一致。"""
    parts = ["Agent线上"]
    if settings.agent_case_retry_enabled:
        parts.append("案情retry")
    return "+".join(parts)


def print_agent_alignment_note() -> None:
    print(
        "\n口径说明: Agent列 = 意图路由 → filter_context → "
        "查条(get_article+检索fallback) / 概念·案情(search_laws)；"
        "案情走 retrieve_fusion + 质量评估 + 二轮补充 + 主题锚点；"
        "混合列 = 全题 retrieve_fusion(rewrite=True)，无路由分流。",
        flush=True,
    )


def hybrid_column_label() -> str:
    """第三列表头。"""
    return (
        f"Cascade池{settings.rrf_pool_k}+Rerank+union"
    )


def print_retrieval_config() -> None:
    parts = [
        f"BM25={'开' if settings.bm25_enabled else '关'}",
        f"Rerank={'开' if settings.rerank_enabled else '关'}",
        f"改写模式={settings.query_rewrite_mode}",
        f"Cascade池={settings.rrf_pool_k}",
        f"路径保底=向量×{settings.path_reserve_vector_top}+BM25×{settings.path_reserve_bm25_top}",
        f"concat检索={'开' if settings.concat_retrieval_enabled else '关'}(权重{settings.concat_rrf_weight})",
        "精排=query_type自适应",
        "选条=案情plain/概念组约束",
        "改写union精排=开",
    ]
    if settings.domain_rrf_boost > 1.0:
        parts.append(f"域加权×{settings.domain_rrf_boost}")
    if settings.bm25_enabled:
        parts.append(f"BM25候选={settings.bm25_candidate_k}")
        parts.append(f"BM25进RRF≤{settings.bm25_rrf_max_entries}条×权重{settings.bm25_rrf_weight}")
    parts.append(f"宽松命中=law_id+acceptable±{EVAL_HIT_TOLERANCE}")
    print("检索配置: " + ", ".join(parts), flush=True)


def load_kb_index() -> tuple[set[tuple[str, str]], dict[tuple[str, int], tuple[str, str]]]:
    """知识库索引：原始键集合 + (法律名, 条号数值) 映射。"""
    collection = get_collection()
    data = collection.get(include=["metadatas"])
    exact: set[tuple[str, str]] = set()
    by_num: dict[tuple[str, int], tuple[str, str]] = {}
    for meta in data["metadatas"]:
        law = meta["law_name"]
        no = meta["article_no"]
        exact.add((law, no))
        num = normalize_article_no(no)
        if num is not None:
            by_num[(law, num)] = (law, no)
    return exact, by_num


LAW_ALIASES = {
    "宪法": "中华人民共和国宪法",
    "中华人民共和国宪法": "中华人民共和国宪法",
    "民法典": "中华人民共和国民法典",
    "中华人民共和国民法典": "中华人民共和国民法典",
    "刑法": "中华人民共和国刑法",
    "中华人民共和国刑法": "中华人民共和国刑法",
    "劳动法": "中华人民共和国劳动法",
    "中华人民共和国劳动法": "中华人民共和国劳动法",
}


def normalize_law_name(name: str) -> str:
    name = name.strip()
    return LAW_ALIASES.get(name, name)


def citation_in_kb(law: str, article_no: str, kb_exact: set[tuple[str, str]], kb_by_num: dict) -> bool:
    law = normalize_law_name(law)
    if (law, article_no) in kb_exact:
        return True
    num = normalize_article_no(article_no)
    return num is not None and (law, num) in kb_by_num


def extract_citations(text: str) -> list[tuple[str, str]]:
    return CITATION_RE.findall(text)


def citation_metrics(
    cited: list[tuple[str, str]],
    expected: list[str],
    kb_exact: set[tuple[str, str]],
    kb_by_num: dict[tuple[str, int], tuple[str, str]],
) -> tuple[float, float, bool]:
    """返回 (召回率, 准确率, 是否幻觉)。"""
    if not expected:
        return 0.0, 0.0, False

    recall = sum(
        1 for exp in expected if any(article_match(no, exp) for _, no in cited)
    ) / len(expected)

    if not cited:
        return recall, 0.0, False

    valid = sum(1 for law, no in cited if citation_in_kb(law, no, kb_exact, kb_by_num))
    precision = valid / len(cited)
    hallucination = valid < len(cited)
    return recall, precision, hallucination


def run_retrieval(query: str, item: dict, top_k: int) -> tuple[list[dict], bool, list[str]]:
    chunks = retrieve(query, top_k=top_k)
    retrieved = [c["article_no"] for c in chunks]
    hit = retrieval_hit(chunks, item)
    return chunks, hit, retrieved


def run_retrieval_fusion(
    question: str,
    item: dict,
    top_k: int,
    *,
    rewrite: bool = True,
) -> tuple[list[dict], bool, list[str], dict]:
    chunks, meta = retrieve_fusion(
        question, None, top_k=top_k, rewrite=rewrite
    )
    retrieved = [c["article_no"] for c in chunks]
    hit = retrieval_hit(chunks, item)
    return chunks, hit, retrieved, meta


def run_agent_retrieval(
    question: str,
    item: dict,
    top_k: int,
) -> tuple[list[dict], bool, list[str], dict]:
    outcome = run_agent_pre_retrieval(question, None)
    chunks = outcome.chunks[:top_k]
    retrieved = [c["article_no"] for c in chunks]
    hit = retrieval_hit(chunks, item)
    meta = {
        "intent": outcome.intent,
        "route_source": outcome.route_source,
        "retrieval_retry": outcome.retrieval_retry,
        "retrieval_retry_reason": outcome.retrieval_retry_reason,
        "tools_run": outcome.tools_run,
        **outcome.retrieve_meta,
    }
    return chunks, hit, retrieved, meta


def run_rewrite_comparison(
    top_k: int | None,
    *,
    include_agent: bool = True,
    eval_file: Path | None = None,
) -> tuple[list[QuestionResult], list[Summary]]:
    k = top_k or settings.top_k
    print("正在加载 Embedding 模型与向量库...", flush=True)
    if not wait_until_ready():
        raise RuntimeError("RAG 组件加载超时")
    if settings.query_rewrite_enabled and not settings.deepseek_api_key:
        raise RuntimeError("Query 改写对比需要配置 DEEPSEEK_API_KEY")
    print("加载完成，开始改写 / 混合 / Agent 线上路径对比评测。", flush=True)
    print_retrieval_config()
    if include_agent:
        print_agent_alignment_note()

    questions = load_questions(eval_file)
    results: list[QuestionResult] = []
    total = len(questions)

    for idx, item in enumerate(questions, start=1):
        qid = item["id"]
        question = item["question"]
        expected = item["expected_articles"]
        print(f"[{idx}/{total}] {qid} {question[:30]}...", flush=True)

        base_q = build_retrieval_query(question, None)
        t0 = time.perf_counter()
        _, hit_base, arts_base = run_retrieval(base_q, item, k)
        ms_base = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        rw_q, source = build_search_query(question, None, rewrite=True)
        _, hit_rw, arts_rw = run_retrieval(rw_q, item, k)
        ms_rw = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        _, hit_fusion, arts_fusion, fusion_meta = run_retrieval_fusion(
            question, item, k, rewrite=True
        )
        ms_fusion = (time.perf_counter() - t2) * 1000
        fusion_mode = fusion_meta.get("fusion_mode", "")
        pool_size = fusion_meta.get("rrf_pool_size", 0)

        print(f"  baseline: {base_q[:40]} -> {'命中' if hit_base else '未命中'}", flush=True)
        print(f"  rewrite:  {rw_q[:40]} -> {'命中' if hit_rw else '未命中'}", flush=True)
        fusion_q = fusion_meta.get("rewrite_query", rw_q)
        extra = f", pool={pool_size}" if pool_size else ""
        print(
            f"  hybrid[{fusion_mode}]: {fusion_q[:40]} -> "
            f"{'命中' if hit_fusion else '未命中'}{extra}",
            flush=True,
        )

        agent_meta: dict = {}
        hit_agent = False
        arts_agent: list[str] = []
        ms_agent = 0.0
        if include_agent:
            t3 = time.perf_counter()
            _, hit_agent, arts_agent, agent_meta = run_agent_retrieval(question, item, k)
            ms_agent = (time.perf_counter() - t3) * 1000
            intent = agent_meta.get("intent", "")
            retry = " retry" if agent_meta.get("retrieval_retry") else ""
            print(
                f"  agent[{intent}{retry}]: "
                f"{'命中' if hit_agent else '未命中'} "
                f"tools={agent_meta.get('tools_run', [])}",
                flush=True,
            )

        results.append(
            QuestionResult(
                id=qid,
                question=question,
                mode="retrieval_baseline",
                recall_at_k=hit_base,
                retrieved_articles=arts_base,
                search_query=base_q,
                query_source="baseline",
                latency_ms=ms_base,
            )
        )
        results.append(
            QuestionResult(
                id=qid,
                question=question,
                mode="retrieval_rewrite",
                recall_at_k=hit_rw,
                retrieved_articles=arts_rw,
                search_query=rw_q,
                query_source=source,
                latency_ms=ms_rw,
            )
        )
        results.append(
            QuestionResult(
                id=qid,
                question=question,
                mode="retrieval_hybrid",
                recall_at_k=hit_fusion,
                retrieved_articles=arts_fusion,
                search_query=fusion_meta.get("baseline_query", base_q)
                + " | "
                + fusion_meta.get("rewrite_query", rw_q),
                query_source=fusion_meta.get("query_source", "cascade"),
                fusion_mode=fusion_meta.get("fusion_mode", ""),
                rrf_pool_size=int(fusion_meta.get("rrf_pool_size") or 0),
                latency_ms=ms_fusion,
            )
        )
        if include_agent:
            results.append(
                QuestionResult(
                    id=qid,
                    question=question,
                    mode="retrieval_agent",
                    recall_at_k=hit_agent,
                    retrieved_articles=arts_agent,
                    search_query=str(agent_meta.get("search_query") or agent_meta.get("rewrite_query") or ""),
                    query_source=agent_meta.get("query_source", "agent"),
                    fusion_mode=agent_meta.get("fusion_mode", ""),
                    rrf_pool_size=int(agent_meta.get("rrf_pool_size") or 0),
                    latency_ms=ms_agent,
                    intent=str(agent_meta.get("intent", "")),
                    route_source=str(agent_meta.get("route_source", "")),
                    retrieval_retry=bool(agent_meta.get("retrieval_retry")),
                    retrieval_retry_reason=str(agent_meta.get("retrieval_retry_reason") or ""),
                    tools_run=list(agent_meta.get("tools_run") or []),
                )
            )
    summaries = summarize_rewrite_comparison(results)
    return results, summaries


def run_agent_only_comparison(
    top_k: int | None,
    *,
    eval_file: Path | None = None,
) -> tuple[list[QuestionResult], list[Summary]]:
    """仅评测 Agent 线上检索路径。"""
    k = top_k or settings.top_k
    print("正在加载 Embedding 模型与向量库...", flush=True)
    if not wait_until_ready():
        raise RuntimeError("RAG 组件加载超时")
    if settings.query_rewrite_enabled and not settings.deepseek_api_key:
        raise RuntimeError("Agent 评测（含 LLM 改写）需要配置 DEEPSEEK_API_KEY")
    print("加载完成，开始 Agent 线上路径检索评测。", flush=True)
    print_retrieval_config()
    print_agent_alignment_note()

    questions = load_questions(eval_file)
    results: list[QuestionResult] = []
    total = len(questions)

    for idx, item in enumerate(questions, start=1):
        qid = item["id"]
        question = item["question"]
        print(f"[{idx}/{total}] {qid} {question[:30]}...", flush=True)
        t0 = time.perf_counter()
        _, hit, arts, meta = run_agent_retrieval(question, item, k)
        ms = (time.perf_counter() - t0) * 1000
        print(
            f"  agent[{meta.get('intent')}]: {'命中' if hit else '未命中'} "
            f"tools={meta.get('tools_run', [])}",
            flush=True,
        )
        results.append(
            QuestionResult(
                id=qid,
                question=question,
                mode="retrieval_agent",
                recall_at_k=hit,
                retrieved_articles=arts,
                search_query=str(meta.get("search_query") or meta.get("rewrite_query") or ""),
                query_source=meta.get("query_source", "agent"),
                fusion_mode=meta.get("fusion_mode", ""),
                rrf_pool_size=int(meta.get("rrf_pool_size") or 0),
                latency_ms=ms,
                intent=str(meta.get("intent", "")),
                route_source=str(meta.get("route_source", "")),
                retrieval_retry=bool(meta.get("retrieval_retry")),
                retrieval_retry_reason=str(meta.get("retrieval_retry_reason") or ""),
                tools_run=list(meta.get("tools_run") or []),
            )
        )

    rows = [r for r in results if r.mode == "retrieval_agent"]
    summary = Summary(
        mode="retrieval_agent",
        count=len(rows),
        recall_at_k=sum(1 for r in rows if r.recall_at_k) / len(rows) if rows else 0.0,
        avg_latency_ms=sum(r.latency_ms for r in rows) / len(rows) if rows else 0.0,
    )
    print_agent_report([summary], results)
    return results, [summary]


def summarize_rewrite_comparison(results: list[QuestionResult]) -> list[Summary]:
    summaries: list[Summary] = []
    for mode in (
        "retrieval_baseline",
        "retrieval_rewrite",
        "retrieval_hybrid",
        "retrieval_agent",
    ):
        rows = [r for r in results if r.mode == mode]
        if not rows:
            continue
        summaries.append(
            Summary(
                mode=mode,
                count=len(rows),
                recall_at_k=sum(1 for r in rows if r.recall_at_k) / len(rows),
                avg_latency_ms=sum(r.latency_ms for r in rows) / len(rows),
            )
        )
    return summaries


def print_agent_report(summaries: list[Summary], results: list[QuestionResult]) -> None:
    agent = next((s for s in summaries if s.mode == "retrieval_agent"), None)
    if not agent:
        return
    print("\n" + "=" * 60)
    print(f"Agent 线上路径检索报告（{agent_column_label()}）")
    print("=" * 60)
    print_retrieval_config()
    print_agent_alignment_note()
    print(f"\n评测集: {EVAL_FILE.name}，共 {agent.count} 题，Top-K={settings.top_k}\n")
    print(f"| Recall@{settings.top_k} | {agent.recall_at_k:.1%} |")
    print(f"| 平均检索时延 | {agent.avg_latency_ms:.0f} ms |")

    retry_rows = [r for r in results if r.mode == "retrieval_agent" and r.retrieval_retry]
    if retry_rows:
        print(f"\n触发案情补充检索: {len(retry_rows)} 题")
        for r in retry_rows[:10]:
            print(f"  {r.id}: {r.retrieval_retry_reason}")
        if len(retry_rows) > 10:
            print(f"  ... 共 {len(retry_rows)} 题")

    by_intent: dict[str, list[QuestionResult]] = {}
    for r in results:
        if r.mode != "retrieval_agent":
            continue
        by_intent.setdefault(r.intent or "unknown", []).append(r)
    if by_intent:
        print("\n按意图 Recall:")
        for intent, rows in sorted(by_intent.items()):
            hit = sum(1 for x in rows if x.recall_at_k) / len(rows)
            print(f"  {intent}: {hit:.1%} ({len(rows)} 题)")


def print_rewrite_report(summaries: list[Summary], results: list[QuestionResult]) -> None:
    hybrid_label = hybrid_column_label()
    agent_label = agent_column_label()
    print("\n" + "=" * 60)
    print("Query 改写 / 混合检索 / Agent 线上路径对比报告")
    print("=" * 60)
    print_retrieval_config()
    if any(s.mode == "retrieval_agent" for s in summaries):
        print_agent_alignment_note()

    base = next((s for s in summaries if s.mode == "retrieval_baseline"), None)
    rw = next((s for s in summaries if s.mode == "retrieval_rewrite"), None)
    fusion = next((s for s in summaries if s.mode == "retrieval_hybrid"), None)
    agent = next((s for s in summaries if s.mode == "retrieval_agent"), None)
    if not base or not rw:
        return

    has_agent = agent is not None
    print(f"\n评测集: {EVAL_FILE.name}，共 {base.count} 题，最终 Top-K={settings.top_k}\n")
    if has_agent:
        print(
            f"| 指标 | 不改写 | Query改写 | {hybrid_label} | {agent_label} |"
        )
        print("|------|--------|---------|" + "-" * max(8, len(hybrid_label)) + "|" + "-" * max(8, len(agent_label)) + "|")
    else:
        print(f"| 指标 | 不改写 (baseline) | Query 改写 | {hybrid_label} |")
        print("|------|-------------------|------------|" + "-" * max(8, len(hybrid_label)) + "|")
    diff_rw = rw.recall_at_k - base.recall_at_k
    diff_rw_str = f"+{diff_rw:.1%}" if diff_rw >= 0 else f"{diff_rw:.1%}"
    fusion_recall = f"{fusion.recall_at_k:.1%}" if fusion else "—"
    agent_recall = f"{agent.recall_at_k:.1%}" if agent else "—"
    if has_agent:
        print(
            f"| Recall@{settings.top_k} | {base.recall_at_k:.1%} | {rw.recall_at_k:.1%} "
            f"({diff_rw_str}) | {fusion_recall} | {agent_recall} |"
        )
    else:
        print(
            f"| Recall@{settings.top_k} | {base.recall_at_k:.1%} | {rw.recall_at_k:.1%} "
            f"({diff_rw_str}) | {fusion_recall} |"
        )
    if fusion:
        diff_fusion = fusion.recall_at_k - base.recall_at_k
        diff_fusion_str = f"+{diff_fusion:.1%}" if diff_fusion >= 0 else f"{diff_fusion:.1%}"
        if has_agent:
            print(
                f"| 平均检索时延 | {base.avg_latency_ms:.0f} ms | {rw.avg_latency_ms:.0f} ms | "
                f"{fusion.avg_latency_ms:.0f} ms | {agent.avg_latency_ms:.0f} ms |"
            )
        else:
            print(
                f"| 平均检索时延 | {base.avg_latency_ms:.0f} ms | {rw.avg_latency_ms:.0f} ms | "
                f"{fusion.avg_latency_ms:.0f} ms |"
            )
        print(f"\n{hybrid_label} 相对 baseline Recall 变化: {diff_fusion_str}")
        if agent:
            diff_agent = agent.recall_at_k - base.recall_at_k
            diff_agent_str = f"+{diff_agent:.1%}" if diff_agent >= 0 else f"{diff_agent:.1%}"
            print(f"{agent_label} 相对 baseline Recall 变化: {diff_agent_str}")

    def diff_cases(mode_a: str, mode_b: str) -> tuple[list[str], list[str]]:
        improved: list[str] = []
        regressed: list[str] = []
        for qid in {r.id for r in results}:
            a = next(r for r in results if r.id == qid and r.mode == mode_a)
            b = next(r for r in results if r.id == qid and r.mode == mode_b)
            if b.recall_at_k and not a.recall_at_k:
                improved.append(qid)
            elif a.recall_at_k and not b.recall_at_k:
                regressed.append(qid)
        return improved, regressed

    rw_improved, rw_regressed = diff_cases("retrieval_baseline", "retrieval_rewrite")
    if rw_improved:
        print(f"\n改写后新命中 ({len(rw_improved)}): {', '.join(rw_improved)}")
    if rw_regressed:
        print(f"改写后丢失命中 ({len(rw_regressed)}): {', '.join(rw_regressed)}")

    if fusion:
        fusion_improved, fusion_regressed = diff_cases("retrieval_baseline", "retrieval_hybrid")
        if fusion_improved:
            print(f"\n{hybrid_label} 新命中 ({len(fusion_improved)}): {', '.join(fusion_improved)}")
        if fusion_regressed:
            print(f"{hybrid_label} 丢失命中 ({len(fusion_regressed)}): {', '.join(fusion_regressed)}")
        vs_rw_improved, vs_rw_regressed = diff_cases("retrieval_rewrite", "retrieval_hybrid")
        if vs_rw_improved:
            print(f"混合相对改写新命中 ({len(vs_rw_improved)}): {', '.join(vs_rw_improved)}")
        if vs_rw_regressed:
            print(f"混合相对改写丢失 ({len(vs_rw_regressed)}): {', '.join(vs_rw_regressed)}")

    if agent and fusion:
        ag_improved, ag_regressed = diff_cases("retrieval_hybrid", "retrieval_agent")
        if ag_improved:
            print(f"\nAgent 相对混合新命中 ({len(ag_improved)}): {', '.join(ag_improved)}")
        if ag_regressed:
            print(f"Agent 相对混合丢失 ({len(ag_regressed)}): {', '.join(ag_regressed)}")
        print("\n线上口径: 以 Agent 列为准；混合列为检索管线消融对照。")


def run_eval(
    retrieval_only: bool = False,
    top_k: int | None = None,
    rewrite: bool | None = None,
    *,
    eval_file: Path | None = None,
) -> tuple[list[QuestionResult], list[Summary]]:
    k = top_k or settings.top_k
    print("正在加载 Embedding 模型与向量库...", flush=True)

    if not wait_until_ready():
        raise RuntimeError("RAG 组件加载超时")
    rewrite_on = settings.query_rewrite_enabled if rewrite is None else rewrite
    print(
        f"加载完成，开始评测（Query 改写: {'开' if rewrite_on else '关'}）。",
        flush=True,
    )
    print_retrieval_config()

    kb_exact, kb_by_num = load_kb_index()
    questions = load_questions(eval_file)
    results: list[QuestionResult] = []
    total = len(questions)

    for idx, item in enumerate(questions, start=1):
        qid = item["id"]
        question = item["question"]
        print(f"[{idx}/{total}] {qid} {question[:30]}...", flush=True)

        t0 = time.perf_counter()
        chunks, meta = retrieve_fusion(
            question, None, top_k=k, rewrite=rewrite_on
        )
        retrieve_ms = (time.perf_counter() - t0) * 1000
        retrieved = [c["article_no"] for c in chunks]
        recall_hit = retrieval_hit(chunks, item)
        search_q = meta.get("search_query", question)
        source = meta.get("query_source", "baseline")
        fusion_mode = meta.get("fusion_mode", "")
        pool_size = meta.get("rrf_pool_size", 0)
        hit_mark = "命中" if recall_hit else "未命中"
        extra = f" pool={pool_size}" if pool_size else ""
        print(f"  -> {fusion_mode or source} {hit_mark}{extra} {retrieved[:3]}...", flush=True)

        results.append(
            QuestionResult(
                id=qid,
                question=question,
                mode="rag_retrieval",
                recall_at_k=recall_hit,
                retrieved_articles=retrieved,
                search_query=search_q,
                query_source=source,
                fusion_mode=fusion_mode,
                rrf_pool_size=int(pool_size or 0),
                latency_ms=retrieve_ms,
            )
        )

        if retrieval_only:
            continue

        if not settings.deepseek_api_key:
            raise RuntimeError("完整评测需要配置 DEEPSEEK_API_KEY")

        print(f"  -> RAG 生成中...", flush=True)

        t1 = time.perf_counter()
        rag_answer = ask_llm(question, chunks)
        rag_ms = (time.perf_counter() - t1) * 1000
        rag_cited = extract_citations(rag_answer)
        r_recall, r_prec, r_hall = citation_metrics(
            rag_cited, item["expected_articles"], kb_exact, kb_by_num
        )

        results.append(
            QuestionResult(
                id=qid,
                question=question,
                mode="rag",
                recall_at_k=recall_hit,
                retrieved_articles=retrieved,
                search_query=search_q,
                query_source=source,
                cited_articles=[f"《{l}》{n}" for l, n in rag_cited],
                citation_recall=r_recall,
                citation_precision=r_prec,
                hallucination=r_hall,
                latency_ms=retrieve_ms + rag_ms,
                answer_preview=rag_answer[:200],
            )
        )

        print(f"  -> 无 RAG 生成中...", flush=True)
        t2 = time.perf_counter()
        no_rag_answer = ask_llm_no_rag(question)
        no_rag_ms = (time.perf_counter() - t2) * 1000
        nr_cited = extract_citations(no_rag_answer)
        nr_recall, nr_prec, nr_hall = citation_metrics(nr_cited, expected, kb_exact, kb_by_num)

        results.append(
            QuestionResult(
                id=qid,
                question=question,
                mode="no_rag",
                cited_articles=[f"《{l}》{n}" for l, n in nr_cited],
                citation_recall=nr_recall,
                citation_precision=nr_prec,
                hallucination=nr_hall,
                latency_ms=no_rag_ms,
                answer_preview=no_rag_answer[:200],
            )
        )

    summaries = summarize(results, retrieval_only)
    return results, summaries


def summarize(results: list[QuestionResult], retrieval_only: bool) -> list[Summary]:
    summaries: list[Summary] = []

    if retrieval_only:
        hits = [r for r in results if r.mode == "rag_retrieval"]
        if hits:
            summaries.append(
                Summary(
                    mode="rag_retrieval",
                    count=len(hits),
                    recall_at_k=sum(1 for r in hits if r.recall_at_k) / len(hits),
                    avg_latency_ms=sum(r.latency_ms for r in hits) / len(hits),
                )
            )
        return summaries

    for mode in ("rag", "no_rag"):
        rows = [r for r in results if r.mode == mode]
        if not rows:
            continue
        rag_rows = [r for r in results if r.mode == "rag"]
        summaries.append(
            Summary(
                mode=mode,
                count=len(rows),
                recall_at_k=(
                    sum(1 for r in rag_rows if r.recall_at_k) / len(rag_rows) if mode == "rag" else None
                ),
                avg_citation_recall=sum(r.citation_recall for r in rows) / len(rows),
                avg_citation_precision=sum(r.citation_precision for r in rows) / len(rows),
                hallucination_rate=sum(1 for r in rows if r.hallucination) / len(rows),
                avg_latency_ms=sum(r.latency_ms for r in rows) / len(rows),
            )
        )
    return summaries


def print_report(summaries: list[Summary], retrieval_only: bool) -> None:
    print("\n" + "=" * 60)
    print("RAG vs 无 RAG 对比评测报告")
    print("=" * 60)

    if retrieval_only:
        s = summaries[0]
        print(f"\n检索评测 (N={s.count}, Top-K={settings.top_k})")
        print(f"  Recall@{settings.top_k}: {s.recall_at_k:.1%}")
        print(f"  平均检索时延: {s.avg_latency_ms:.0f} ms")
        return

    rag = next((s for s in summaries if s.mode == "rag"), None)
    no_rag = next((s for s in summaries if s.mode == "no_rag"), None)
    if not rag or not no_rag:
        return

    print(f"\n评测集: {EVAL_FILE.name}，共 {rag.count} 题，Top-K={settings.top_k}\n")
    print("| 指标 | RAG | 无 RAG | 提升 |")
    print("|------|-----|--------|------|")

    def delta(a: float, b: float) -> str:
        diff = a - b
        return f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"

    print(f"| Recall@{settings.top_k}（检索命中） | {rag.recall_at_k:.1%} | — | — |")
    print(
        f"| 法条引用召回率 | {rag.avg_citation_recall:.1%} | {no_rag.avg_citation_recall:.1%} | "
        f"{delta(rag.avg_citation_recall, no_rag.avg_citation_recall)} |"
    )
    print(
        f"| 法条引用准确率 | {rag.avg_citation_precision:.1%} | {no_rag.avg_citation_precision:.1%} | "
        f"{delta(rag.avg_citation_precision, no_rag.avg_citation_precision)} |"
    )
    print(
        f"| 幻觉法条率 | {rag.hallucination_rate:.1%} | {no_rag.hallucination_rate:.1%} | "
        f"{delta(no_rag.hallucination_rate, rag.hallucination_rate)} |"
    )
    print(
        f"| 平均响应时延 | {rag.avg_latency_ms:.0f} ms | {no_rag.avg_latency_ms:.0f} ms | "
        f"+{rag.avg_latency_ms - no_rag.avg_latency_ms:.0f} ms |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG vs 无 RAG 对比评测")
    parser.add_argument("--retrieval-only", action="store_true", help="仅评测检索，不调用 LLM")
    parser.add_argument(
        "--compare-rewrite",
        action="store_true",
        help="对比 baseline / 改写 / 混合 / Agent 线上路径（需 DEEPSEEK_API_KEY）",
    )
    parser.add_argument(
        "--compare-agent",
        action="store_true",
        help="仅评测 Agent 线上检索路径（与 /api/ask 一致）",
    )
    parser.add_argument(
        "--no-agent-column",
        action="store_true",
        help="与 --compare-rewrite 联用时跳过 Agent 第四列",
    )
    parser.add_argument("--no-rewrite", action="store_true", help="关闭 Query 改写")
    parser.add_argument("--output", type=str, default="", help="保存 JSON 结果路径")
    parser.add_argument("--top-k", type=int, default=None, help="检索条数，默认读取配置")
    parser.add_argument(
        "--eval-file",
        type=str,
        default="",
        help="评测集 YAML（默认 data/eval_questions_verified.yaml）",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="Recall 门禁阈值（0–1）；未达标时 exit 1",
    )
    parser.add_argument(
        "--gate-mode",
        type=str,
        default="retrieval_agent",
        help="门禁统计列：retrieval_agent / rag_retrieval / retrieval_hybrid 等",
    )
    args = parser.parse_args()

    rewrite = False if args.no_rewrite else None
    eval_file = Path(args.eval_file) if args.eval_file else None
    if eval_file and not eval_file.is_absolute():
        eval_file = BACKEND / eval_file
    eval_path = eval_file or EVAL_FILE

    if args.compare_agent:
        results, summaries = run_agent_only_comparison(args.top_k, eval_file=eval_path)
    elif args.compare_rewrite:
        results, summaries = run_rewrite_comparison(
            args.top_k,
            include_agent=not args.no_agent_column,
            eval_file=eval_path,
        )
        print_rewrite_report(summaries, results)
    else:
        results, summaries = run_eval(
            retrieval_only=args.retrieval_only,
            top_k=args.top_k,
            rewrite=rewrite,
            eval_file=eval_path,
        )
        print_report(summaries, args.retrieval_only)

    if args.min_recall is not None:
        enforce_recall_gate(
            summaries,
            gate_mode=args.gate_mode,
            min_recall=args.min_recall,
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "top_k": args.top_k or settings.top_k,
            "compare_rewrite": args.compare_rewrite,
            "compare_agent": args.compare_agent,
            "agent_column_included": args.compare_rewrite and not args.no_agent_column,
            "eval_alignment": "retrieval_agent uses agent.pre_retrieval.run_agent_pre_retrieval",
            "query_rewrite_enabled": settings.query_rewrite_enabled if rewrite is None else rewrite,
            "query_rewrite_mode": settings.query_rewrite_mode,
            "retrieval": "cascade_union",
            "rerank_enabled": settings.rerank_enabled,
            "rerank_model": settings.rerank_model,
            "bm25_enabled": settings.bm25_enabled,
            "rrf_pool_k": settings.rrf_pool_k,
            "concat_retrieval_enabled": settings.concat_retrieval_enabled,
            "concat_rrf_weight": settings.concat_rrf_weight,
            "hybrid_fusion_label": hybrid_column_label(),
            "eval_hit_tolerance": EVAL_HIT_TOLERANCE,
            "eval_hit_rule": "law_id + expected + acceptable + expected±tolerance",
            "summaries": [asdict(s) for s in summaries],
            "details": [asdict(r) for r in results],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
