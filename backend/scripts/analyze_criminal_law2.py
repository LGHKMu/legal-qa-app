"""深入分析刑法条号与缺失情况。"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def cn2int(article_no: str) -> int | None:
    s = re.sub(r"第|条|之.*", "", article_no)
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if s.startswith("十") and len(s) == 2:
        return 10 + CN.get(s[1], 0)
    if "百" in s:
        parts = s.split("百", 1)
        b = CN.get(parts[0], 1) if parts[0] else 1
        rest = parts[1]
        if not rest:
            return b * 100
        if rest == "十":
            return b * 100 + 10
        if rest.startswith("十"):
            return b * 100 + 10 + CN.get(rest[1], 0)
        if "十" in rest:
            t, p = rest.split("十", 1)
            return b * 100 + (CN.get(t, 0) * 10 if t else 10) + CN.get(p, 0)
        return b * 100 + CN.get(rest, 0)
    if "十" in s:
        a, b = s.split("十", 1)
        return (CN.get(a, 0) or 1) * 10 + CN.get(b, 0)
    return CN.get(s)


def main() -> None:
    arts = json.loads(Path("data/criminal_law_articles.json").read_text(encoding="utf-8"))

    by_num: dict[int, list[dict]] = defaultdict(list)
    for a in arts:
        n = cn2int(a["no"])
        if n is not None:
            by_num[n].append(a)

    num_dup = {k: v for k, v in by_num.items() if len(v) > 1}
    extra = sum(len(v) - 1 for v in num_dup.values())
    print(f"条号可解析为数字: {sum(len(v) for v in by_num.values())}")
    print(f"不同条号(数字): {len(by_num)}")
    print(f"同一数字多条文本(去重前碎片): {len(num_dup)} 组, 多出 {extra} 条")
    print(f"条号范围: 第{min(by_num)}条 — 第{max(by_num)}条")

    missing = [i for i in range(1, max(by_num) + 1) if i not in by_num]
    print(f"1..{max(by_num)} 缺失: {len(missing)} 条")
    print(f"缺失示例: {missing[:20]} ... {missing[-10:]}")

    trunc = [
        a
        for a in arts
        if len(a["text"]) < 25
        or a["text"].endswith(("除", "的，", "、", "（", "之"))
        or a["text"] in {"的", "款", "项"}
    ]
    print(f"疑似被误切/截断: {len(trunc)} 条")

    # 273 vs 262: 多出来的主要是重复条号碎片 + 目录/引用误匹配
    print(f"\n网站显示 273 = 去重后唯一「第X条」字符串数")
    print(f"实际完整法条约 452+(修正案) 条; 当前 HTML 最高只到第{max(by_num)}条且大量缺失")

    out = Path("data/criminal_law_report.txt")
    lines = []
    for n in sorted(by_num):
        for a in by_num[n]:
            lines.append(f"第{n}条\t{a['no']}\t{len(a['text'])}\t{a['text'][:80]}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"明细: {out}")


if __name__ == "__main__":
    main()
