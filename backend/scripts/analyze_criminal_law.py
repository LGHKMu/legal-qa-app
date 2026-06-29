"""分析刑法条数解析情况。"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup
from parser import (
    ARTICLE_SPLIT,
    _extract_html_text,
    _normalize,
    dedupe_articles,
    parse_html,
    parse_plain_text,
)

html = Path("data/raw/criminal_law.html").read_text(encoding="utf-8", errors="replace")
after = parse_html(
    html, law_id="criminal_law", law_name="中华人民共和国刑法", source_url=""
)

soup = BeautifulSoup(html, "html.parser")
for tag in soup(["script", "style", "nav", "header", "footer"]):
    tag.decompose()
text = _extract_html_text(soup) or _normalize(html)
before = parse_plain_text(
    text, law_id="criminal_law", law_name="中华人民共和国刑法", source_url=""
)

print("before dedupe:", len(before))
print("after dedupe:", len(after))

c = Counter(a.article_no for a in before)
dups = {k: v for k, v in c.items() if v > 1}
print("duplicate article_no:", len(dups))
for k, v in sorted(dups.items(), key=lambda x: -x[1])[:20]:
    print(f"  {k}: {v}x")

nos = sorted([a.article_no for a in after], key=lambda x: (len(x), x))
print("unique article_no:", len(nos))

std = [n for n in nos if re.fullmatch(r"第\d+条", n)]
cn = [
    n
    for n in nos
    if re.fullmatch(r"第[零〇一二三四五六七八九十百千万]+条", n)
    and not re.fullmatch(r"第\d+条", n)
]
other = [n for n in nos if n not in std and n not in cn]
print("digit format:", len(std), "chinese num:", len(cn), "other:", len(other))
if other:
    print("other:", other)

digits = []
for n in nos:
    m = re.fullmatch(r"第(\d+)条", n)
    if m:
        digits.append(int(m.group(1)))
if digits:
    print("max digit article:", max(digits))
    missing = [i for i in range(1, max(digits) + 1) if f"第{i}条" not in nos]
    print("missing 1..max:", len(missing))

all_matches = ARTICLE_SPLIT.findall(text)
print("raw regex 第X条 matches:", len(all_matches))
print("unique match strings:", len(set(m.strip() for m in all_matches if m.strip())))

short = [a for a in after if len(a.text) < 40]
print("short body (<40 chars):", len(short))
for a in short[:12]:
    print(f"  {a.article_no}: {a.text[:80]!r}")

# 条文引用误切：正文以「依照/根据/按照」开头
ref_like = [a for a in after if re.match(r"^[，,、]?依照第", a.text) or "条规定" in a.text[:20]]
print("possible reference splits:", len(ref_like))
for a in ref_like[:8]:
    print(f"  {a.article_no}: {a.text[:80]!r}")
