"""回答中的法条引用抽取与校验。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from config import DATA_DIR, LAWS_YAML

CITATION_RE = re.compile(r"《([^》]{2,40})》\s*(第[零〇一二三四五六七八九十百千万\d]+条)")
LEGAL_BASIS_MARK = "【法律依据】"

CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

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


@dataclass
class InvalidCitation:
    law: str
    article_no: str
    reason: str


@dataclass
class VerifyResult:
    passed: bool
    cited: list[tuple[str, str]] = field(default_factory=list)
    invalid: list[InvalidCitation] = field(default_factory=list)
    warnings: list[InvalidCitation] = field(default_factory=list)
    precision: float = 1.0
    hallucination: bool = False
    cited_count: int = 0
    invalid_count: int = 0

    def to_trace_output(self) -> dict:
        return {
            "passed": self.passed,
            "cited_count": self.cited_count,
            "invalid_count": self.invalid_count,
            "precision": round(self.precision, 3),
            "hallucination": self.hallucination,
            "invalid": [
                {"law": i.law, "article_no": i.article_no, "reason": i.reason}
                for i in self.invalid
            ],
            "warnings": [
                {"law": w.law, "article_no": w.article_no, "reason": w.reason}
                for w in self.warnings
            ],
        }


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


def normalize_law_name(name: str) -> str:
    name = name.strip()
    return LAW_ALIASES.get(name, name)


def extract_citations(text: str, *, legal_basis_only: bool = True) -> list[tuple[str, str]]:
    """从回答中抽取《法律名》第X条；默认只解析【法律依据】段。"""
    source = text
    if legal_basis_only and LEGAL_BASIS_MARK in text:
        source = text.split(LEGAL_BASIS_MARK, 1)[1]
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for law, article_no in CITATION_RE.findall(source):
        key = (normalize_law_name(law), article_no)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


@lru_cache(maxsize=1)
def load_kb_index() -> tuple[set[tuple[str, str]], dict[tuple[str, int], tuple[str, str]]]:
    """从 data/parsed/*.json 构建知识库索引。"""
    exact: set[tuple[str, str]] = set()
    by_num: dict[tuple[str, int], tuple[str, str]] = {}
    parsed_dir = DATA_DIR / "parsed"
    if parsed_dir.is_dir():
        for path in sorted(parsed_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            law_name = data.get("law_name", "")
            for art in data.get("articles", []):
                no = art.get("article_no", "")
                if not law_name or not no:
                    continue
                exact.add((law_name, no))
                num = normalize_article_no(no)
                if num is not None:
                    by_num[(law_name, num)] = (law_name, no)
    return exact, by_num


def citation_in_kb(
    law: str,
    article_no: str,
    kb_exact: set[tuple[str, str]],
    kb_by_num: dict[tuple[str, int], tuple[str, str]],
) -> bool:
    law = normalize_law_name(law)
    if (law, article_no) in kb_exact:
        return True
    num = normalize_article_no(article_no)
    return num is not None and (law, num) in kb_by_num


def citation_in_chunks(law: str, article_no: str, chunks: list[dict]) -> bool:
    law = normalize_law_name(law)
    for chunk in chunks:
        chunk_law = normalize_law_name(chunk.get("law_name", ""))
        if chunk_law != law:
            continue
        if article_match(chunk.get("article_no", ""), article_no):
            return True
    return False


def select_chunks_cited_in_answer(chunks: list[dict], answer: str) -> list[dict]:
    """从回答【法律依据】中抽取引用，并映射回检索 chunks（保持引用顺序）。"""
    cited = extract_citations(answer)
    if not cited:
        return []

    matched: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for law, article_no in cited:
        for chunk in chunks:
            chunk_law = normalize_law_name(chunk.get("law_name", ""))
            chunk_no = chunk.get("article_no", "")
            if chunk_law != law or not article_match(chunk_no, article_no):
                continue
            key = (chunk_law, chunk_no)
            if key not in seen:
                seen.add(key)
                matched.append(chunk)
            break
    return matched


def verify_citations(answer: str, chunks: list[dict]) -> VerifyResult:
    """校验回答引用：不在 KB 为 invalid；在 KB 但不在本次 chunks 为 warning。"""
    cited = extract_citations(answer)
    kb_exact, kb_by_num = load_kb_index()

    invalid: list[InvalidCitation] = []
    warnings: list[InvalidCitation] = []

    for law, article_no in cited:
        if not citation_in_kb(law, article_no, kb_exact, kb_by_num):
            invalid.append(InvalidCitation(law, article_no, "not_in_kb"))
        elif chunks and not citation_in_chunks(law, article_no, chunks):
            warnings.append(InvalidCitation(law, article_no, "not_in_retrieved_chunks"))

    cited_count = len(cited)
    invalid_count = len(invalid)
    precision = 1.0 if cited_count == 0 else (cited_count - invalid_count) / cited_count
    hallucination = invalid_count > 0

    return VerifyResult(
        passed=invalid_count == 0,
        cited=cited,
        invalid=invalid,
        warnings=warnings,
        precision=precision,
        hallucination=hallucination,
        cited_count=cited_count,
        invalid_count=invalid_count,
    )
