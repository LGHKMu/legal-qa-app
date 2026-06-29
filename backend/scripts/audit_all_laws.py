"""检查各法律解析质量：条数、残缺条、条号缺口。"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from config import LAWS_YAML
from fetcher import parse_law

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


def is_fragment(text: str) -> bool:
    t = text.strip()
    if len(t) < 15:
        return True
    if t.startswith(("的规定", "依照", "依本法", "犯前款", "本条", "前款")):
        return True
    if t.endswith(("除", "总则", "分则", "之", "、", "（", "的，")):
        return True
    return False


def audit_law(law: dict) -> dict:
    raw_path = Path("data/raw") / law["raw_file"]
    if raw_path.suffix.lower() == ".pdf":
        txt = raw_path.with_suffix(".txt")
        raw_path = txt if txt.exists() else raw_path
    if not raw_path.exists():
        return {"id": law["id"], "error": f"missing {raw_path}"}
    raw = raw_path.read_text(encoding="utf-8", errors="replace")
    if law["raw_file"].endswith(".pdf") and raw_path.suffix.lower() == ".pdf":
        from parser import clean_pdf_text

        raw = clean_pdf_text(raw)
    arts = parse_law(law, raw)
    nums = [cn2int(a.article_no) for a in arts]
    nums_ok = [n for n in nums if n is not None]
    frags = [a for a in arts if is_fragment(a.text)]
    dup = {k: v for k, v in Counter(a.article_no for a in arts).items() if v > 1}
    missing = []
    if nums_ok:
        missing = [i for i in range(1, max(nums_ok) + 1) if i not in nums_ok]
    return {
        "id": law["id"],
        "name": law["name"],
        "count": len(arts),
        "max_no": max(nums_ok) if nums_ok else None,
        "missing_in_range": len(missing),
        "fragments": len(frags),
        "dup_article_no": len(dup),
        "short_samples": [(a.article_no, a.text[:50]) for a in frags[:3]],
        "last": arts[-1].article_no if arts else None,
    }


def main() -> None:
    laws = yaml.safe_load(LAWS_YAML.read_text(encoding="utf-8"))["laws"]
    for law in laws:
        if not law.get("enabled", True):
            continue
        r = audit_law(law)
        print(f"\n=== {r.get('name', r['id'])} ===")
        if "error" in r:
            print("ERROR:", r["error"])
            continue
        print(f"  条数: {r['count']}")
        print(f"  最大条号: {r['max_no']} ({r['last']})")
        print(f"  1..max 缺失: {r['missing_in_range']}")
        print(f"  疑似残片: {r['fragments']}")
        print(f"  重复条号: {r['dup_article_no']}")
        if r["short_samples"]:
            for no, txt in r["short_samples"]:
                print(f"    {no}: {txt!r}")


if __name__ == "__main__":
    main()
