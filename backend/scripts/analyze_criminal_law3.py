"""统计刑法解析质量：完整条 vs 碎片条。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_criminal_law2 import cn2int

arts = json.loads(Path("data/criminal_law_articles.json").read_text(encoding="utf-8"))

FRAG_PATTERNS = (
    r"^的规定",
    r"^依照第",
    r"^依第",
    r"^本条",
    r"^前款",
    r"^犯前款",
    r"^对前款",
    r"^除$",
    r"^的$",
    r"^款$",
    r"^项$",
    r"^之$",
)


def is_fragment(a: dict) -> bool:
    t = a["text"].strip()
    if len(t) < 20:
        return True
    if any(re.match(p, t) for p in FRAG_PATTERNS):
        return True
    if t.endswith(("除", "总则", "分则", "之", "、", "（", "，", "的，")):
        return True
    if "本章上述" in t and len(t) < 40:
        return True
    return False


frag = [a for a in arts if is_fragment(a)]
good = [a for a in arts if not is_fragment(a)]

by_num: dict[int, dict] = {}
for a in good:
    n = cn2int(a["no"])
    if n and (n not in by_num or len(a["text"]) > len(by_num[n]["text"])):
        by_num[n] = a

print("total parsed (deduped strings):", len(arts))
print("fragment / mis-split:", len(frag))
print("looks complete:", len(good))
print("distinct numeric (best per number):", len(by_num))
print("max article number:", max(by_num) if by_num else None)

# show fragments
print("\n--- fragments (first 20) ---")
for a in frag[:20]:
    print(f"{a['no']}\t{len(a['text'])}\t{a['text'][:70]}")

Path("data/criminal_law_fragments.json").write_text(
    json.dumps(frag, ensure_ascii=False, indent=2), encoding="utf-8"
)
