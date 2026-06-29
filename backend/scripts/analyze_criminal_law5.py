"""统计仅由误切产生的条号。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_criminal_law3 import is_fragment
from scripts.analyze_criminal_law2 import cn2int

arts = json.loads(Path("data/criminal_law_articles.json").read_text(encoding="utf-8"))

by_num: dict[int, list] = {}
for a in arts:
    n = cn2int(a["no"])
    if n:
        by_num.setdefault(n, []).append(a)

only_frag_nums = []
mixed = []
good_only = []
for n, items in by_num.items():
    fr = [a for a in items if is_fragment(a)]
    gd = [a for a in items if not is_fragment(a)]
    if gd and fr:
        mixed.append(n)
    elif gd:
        good_only.append(n)
    else:
        only_frag_nums.append(n)

print("only fragment nums:", len(only_frag_nums))
print("mixed:", len(mixed))
print("good only:", len(good_only))
print("total numeric:", len(by_num))
print("273 strings; good_only nums:", len(good_only), "+ mixed with at least 1 good:", len(mixed))
print("phantom-only nums sample:", only_frag_nums[:20])

# strings that are fragments but use a unique article_no with no good sibling
phantom_strings = [a for a in arts if is_fragment(a) and cn2int(a["no"]) in only_frag_nums]
print("fragment entries on phantom-only nums:", len(phantom_strings))

# extra beyond good_only count
print("273 - good_only", 273 - len(good_only), "extra strings from mixed/duplicate labels")
