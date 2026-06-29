"""检索 Query 改写：一阶段改写与方案七两阶段要素抽取。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field

from config import settings
from llm import get_client
from prompts import EXTRACT_SYSTEM, REWRITE_SYSTEM

logger = logging.getLogger(__name__)

ARTICLE_NO_RE = re.compile(r"第[零〇一二三四五六七八九十百千万\d]+条")
VALID_DOMAINS = frozenset({"宪法", "民法典", "刑法", "劳动法"})

DOMAIN_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("加班", "工资", "劳动", "辞退", "工伤", "试用期", "劳动合同", "仲裁"), "劳动法"),
    (("杀人", "盗窃", "诈骗", "犯罪", "判刑", "正当防卫", "交通肇事", "过失"), "刑法"),
    (("离婚", "结婚", "继承", "合同", "侵权", "隐私", "物权", "借款"), "民法典"),
    (("选举", "言论", "宪法", "基本权利", "人身自由", "受教育"), "宪法"),
)

DOMAIN_TO_LAW_ID: dict[str, str] = {
    "宪法": "constitution",
    "民法典": "civil_code",
    "刑法": "criminal_law",
    "劳动法": "labor_law",
}

CASE_MARKERS: tuple[str, ...] = (
    "法院",
    "仲裁",
    "判决",
    "案例",
    "裁决",
    "检察院",
    "一案",
    "案情",
    "原告",
    "被告",
)


@dataclass
class LegalElements:
    domains: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    acts: list[str] = field(default_factory=list)
    rights: list[str] = field(default_factory=list)
    legal_concepts: list[str] = field(default_factory=list)
    query_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalContext:
    """检索侧 Query 理解结果（工业 cascade 路由用）。"""

    question: str
    rewrite_q: str | None = None
    rewrite_source: str = "baseline"
    elements: LegalElements | None = None
    inferred_law_id: str | None = None
    domain_confidence: float = 0.0
    query_type: str = "concept"  # statute | case | concept


def domain_to_law_id(domain: str) -> str | None:
    return DOMAIN_TO_LAW_ID.get(domain.strip())


def classify_query_type(question: str) -> str:
    """案情 / 条号 / 概念 分流，用于精排 query 与选条策略。"""
    if is_article_lookup(question):
        return "statute"
    q = question.strip()
    if len(q) > 50 or any(m in q for m in CASE_MARKERS):
        return "case"
    return "concept"


def _infer_domain_from_elements(elements: LegalElements | None) -> tuple[str | None, float]:
    if elements is None or not elements.domains:
        return None, 0.0
    domains = elements.domains
    if len(domains) == 1:
        return domain_to_law_id(domains[0]), 1.0
    law_ids = [domain_to_law_id(d) for d in domains]
    law_ids = [x for x in law_ids if x]
    if len(law_ids) == 1:
        return law_ids[0], 0.85
    return None, 0.5


def _infer_domain_from_rules(question: str) -> tuple[str | None, float]:
    matched: list[str] = []
    for keywords, law in DOMAIN_HINTS:
        if any(k in question for k in keywords):
            matched.append(law)
    matched = list(dict.fromkeys(matched))
    if len(matched) == 1:
        return domain_to_law_id(matched[0]), 0.8
    if len(matched) > 1:
        return None, 0.4
    return None, 0.0


def infer_retrieval_context(
    question: str,
    rewrite_q: str | None,
    *,
    source: str = "baseline",
    elements: LegalElements | None = None,
) -> RetrievalContext:
    """综合要素抽取与规则，产出软路由与 query_type。"""
    law_id, conf = _infer_domain_from_elements(elements)
    if law_id is None and conf < 0.7:
        rule_law, rule_conf = _infer_domain_from_rules(question)
        if rule_conf > conf:
            law_id, conf = rule_law, rule_conf
    return RetrievalContext(
        question=question.strip(),
        rewrite_q=rewrite_q,
        rewrite_source=source,
        elements=elements,
        inferred_law_id=law_id,
        domain_confidence=conf,
        query_type=classify_query_type(question),
    )


def is_article_lookup(question: str) -> bool:
    """问题含明确条号时，跳过 LLM 改写。"""
    return bool(ARTICLE_NO_RE.search(question))


def rule_domain_hints(question: str) -> str:
    laws: list[str] = []
    for keywords, law in DOMAIN_HINTS:
        if any(k in question for k in keywords):
            laws.append(law)
    return " ".join(dict.fromkeys(laws))


def _format_rewrite_input(question: str, history: list[dict] | None) -> str:
    parts: list[str] = []
    if history:
        parts.append("【相关对话历史】")
        for h in history[-4:]:
            role = h.get("role", "user")
            content = (h.get("content") or "")[:200]
            parts.append(f"{role}: {content}")
    parts.append(f"\n【当前问题】\n{question}")
    hints = rule_domain_hints(question)
    if hints:
        parts.append(f"\n【法律领域提示】{hints}")
    return "\n".join(parts)


def _parse_json_object(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def _clean_str_list(value: object, *, limit: int = 3) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s and s not in items:
            items.append(s)
        if len(items) >= limit:
            break
    return items


def normalize_elements(raw: dict) -> LegalElements:
    domains = [d for d in _clean_str_list(raw.get("domains"), limit=2) if d in VALID_DOMAINS]
    keywords = _clean_str_list(raw.get("query_keywords"), limit=6)
    return LegalElements(
        domains=domains,
        topics=_clean_str_list(raw.get("topics")),
        acts=_clean_str_list(raw.get("acts")),
        rights=_clean_str_list(raw.get("rights")),
        legal_concepts=_clean_str_list(raw.get("legal_concepts")),
        query_keywords=keywords,
    )


def _pick_primary_domain(elements: LegalElements) -> str | None:
    """选 1 个主 domain 追加到 query 末尾。"""
    constitutional = {"言论自由", "选举权", "人身自由", "集会", "游行", "示威", "通信自由"}
    pool = set(elements.query_keywords + elements.rights + elements.topics + elements.legal_concepts)
    if pool & constitutional and "宪法" in elements.domains:
        return "宪法"
    return elements.domains[0] if elements.domains else None


def build_query_from_elements(elements: LegalElements, *, max_len: int = 30) -> str:
    """阶段2：规则拼接检索 query（短、准，与一阶段 30 字上限一致）。"""
    parts: list[str] = []
    seen: set[str] = set()

    def add(item: str) -> None:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            parts.append(item)

    # 只用 query_keywords（最多 4 个），避免 topics/acts 重复堆叠导致截断丢词
    for kw in elements.query_keywords[:4]:
        add(kw)

    domain = _pick_primary_domain(elements)
    if domain:
        add(domain)

    if not parts:
        for kw in (elements.rights + elements.topics)[:2]:
            add(kw)
        if elements.domains:
            add(elements.domains[0])

    return " ".join(parts)[:max_len].strip()


def extract_legal_elements(
    question: str,
    history: list[dict] | None = None,
) -> LegalElements | None:
    """阶段1：LLM 抽取法律要素 JSON。"""
    if not settings.deepseek_api_key:
        logger.warning("未配置 DEEPSEEK_API_KEY，跳过法律要素抽取")
        return None

    client = get_client()
    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": _format_rewrite_input(question, history)},
            ],
            temperature=0,
            max_tokens=settings.query_extract_max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        raw = _parse_json_object(text)
        if not raw:
            logger.warning("法律要素 JSON 解析失败: %s", text[:120])
            return None
        elements = normalize_elements(raw)
        if not elements.query_keywords and not elements.domains:
            hints = rule_domain_hints(question)
            if hints:
                elements.domains = hints.split()
        return elements
    except Exception as exc:
        logger.warning("法律要素抽取失败: %s", exc)
        return None


def rewrite_query(question: str, history: list[dict] | None = None) -> str | None:
    """一阶段 LLM 改写检索 query；失败返回 None。"""
    if is_article_lookup(question):
        return question.strip()

    if not settings.deepseek_api_key:
        logger.warning("未配置 DEEPSEEK_API_KEY，跳过 Query 改写")
        return None

    client = get_client()
    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": _format_rewrite_input(question, history)},
            ],
            temperature=0,
            max_tokens=settings.query_rewrite_max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return None
        line = text.splitlines()[0].strip().strip('"').strip("'").strip("「」")
        if not line or len(line) < 2:
            return None
        return line[:80]
    except Exception as exc:
        logger.warning("Query 改写失败: %s", exc)
        return None


def rewrite_query_two_stage(
    question: str,
    history: list[dict] | None = None,
) -> tuple[str | None, LegalElements | None]:
    """方案七：要素抽取 + 规则拼接；失败返回 (None, None)。"""
    if is_article_lookup(question):
        return question.strip(), None

    elements = extract_legal_elements(question, history)
    if elements is None:
        return None, None

    query = build_query_from_elements(elements)
    if query:
        logger.debug("两阶段改写: %s -> %s (%s)", question[:40], query, elements.to_dict())
        return query, elements

    return None, elements


def _history_key(history: list[dict] | None) -> str:
    if not history:
        return ""
    return "|".join(
        f"{h.get('role', 'user')}:{(h.get('content') or '')[:200]}"
        for h in history[-4:]
    )


# 同一问题在同一进程内复用改写结果，避免 compare-modes 多次调 LLM 导致 query 不一致
_rewrite_cache: dict[tuple[str, str, str], tuple[str | None, str, LegalElements | None]] = {}


def clear_rewrite_cache() -> None:
    _rewrite_cache.clear()


def rewrite_for_search(
    question: str,
    history: list[dict] | None = None,
) -> tuple[str | None, str, LegalElements | None]:
    """统一改写入口。

    返回 (search_query, source, elements)。
    source: two_stage | rewrite | article_lookup | baseline
    """
    if is_article_lookup(question):
        return question.strip(), "article_lookup", None

    mode = (settings.query_rewrite_mode or "two_stage").strip().lower()
    cache_key = (question.strip(), _history_key(history), mode)
    if cache_key in _rewrite_cache:
        return _rewrite_cache[cache_key]

    result = _rewrite_for_search_impl(question, history)
    _rewrite_cache[cache_key] = result
    return result


def _rewrite_for_search_impl(
    question: str,
    history: list[dict] | None,
) -> tuple[str | None, str, LegalElements | None]:
    query, elements = rewrite_query_two_stage(question, history)
    if query:
        return query, "two_stage", elements
    fallback = rewrite_query(question, history)
    if fallback:
        return fallback, "rewrite", elements
    return None, "baseline", elements
