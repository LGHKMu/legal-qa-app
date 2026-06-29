"""找出因正文引用「第X条」误切产生的多余条。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_criminal_law2 import cn2int

arts = json.loads(Path("data/criminal_law_articles.json").read_text(encoding="utf-8"))

# 按数字分组
by_num: dict[int, list] = {}
for a in arts:
    n = cn2int(a["no"])
    if n:
        by_num.setdefault(n, []).append(a)

multi = {k: v for k, v in by_num.items() if len(v) > 1}
print(f"同一法条号被切成多段: {len(multi)} 个条号")
extra = sum(len(v) - 1 for v in multi.values())
print(f"因此多出的条目: {extra}")

for k, v in sorted(multi.items()):
    print(f"\n第{k}条 ({len(v)} 段):")
    for a in sorted(v, key=lambda x: -len(x["text"])):
        print(f"  len={len(a['text']):3d}  {a['text'][:90]}")

# 明显是引用残片、却独占条号的
ref_frag = []
for a in arts:
    t = a["text"].strip()
    if (
        t.startswith(("的规定", "依照", "依本法", "犯本节", "犯前款", "对单位"))
        or t in {"、", "至", "的", "款", "项", "之"}
        or (len(t) <= 15 and ("规定" in t or t.endswith(("除", "的，", "、"))))
    ):
        ref_frag.append(a)

print(f"\n明显引用残片条目: {len(ref_frag)}")
for a in ref_frag:
    print(f"  {a['no']}: {a['text'][:60]!r}")

print(f"\n273 - {len(ref_frag)} = {273 - len(ref_frag)} (若减去引用残片)")
