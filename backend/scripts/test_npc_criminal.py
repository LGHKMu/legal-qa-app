"""Quick test NPC criminal law parsing."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import parse_html, dedupe_articles, CN_NUM, ARTICLE_SPLIT

html = Path(r"C:\Users\L\.cursor\projects\d-test\uploads\t20190522_46193-0.html").read_text(encoding="utf-8")
# strip markdown header lines if present
if html.startswith("Source URL:"):
    html = "\n".join(line for line in html.splitlines() if not line.startswith("Source URL:") and not line.startswith("Title:"))

arts = parse_html(html, law_id="criminal_law", law_name="中华人民共和国刑法", source_url="")
print("current parser:", len(arts))

# line-start articles only
from bs4 import BeautifulSoup
from parser import _extract_html_text, _normalize, parse_plain_text

# treat as plain text after stripping front matter
text = html
if "# 中华人民共和国刑法" in text:
    idx = text.index("第一编 总 则") if "第一编 总 则" in text else text.index("第一编 总则")
    text = text[idx:]

LINE_ART = re.compile(rf"^(第{CN_NUM}条)\s*(.*)", re.M)
matches = list(LINE_ART.finditer(text))
print("line-start articles:", len(matches))
if matches:
    print("first:", matches[0].group(1))
    print("last:", matches[-1].group(1))

# cross-ref in middle of line
all_split = ARTICLE_SPLIT.findall(text)
print("fulltext split count:", len(all_split))
print("extra from cross-refs:", len(all_split) - len(matches))
