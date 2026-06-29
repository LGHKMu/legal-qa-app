"""Compare line-based vs legacy parser on all cached laws."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from config import LAWS_YAML
from fetcher import parse_law

for law in yaml.safe_load(LAWS_YAML.read_text(encoding="utf-8"))["laws"]:
    if not law.get("enabled", True):
        continue
    raw_path = Path("data/raw") / law["raw_file"]
    if raw_path.suffix == ".pdf":
        raw_path = raw_path.with_suffix(".txt")
    if not raw_path.exists() and law["id"] == "criminal_law":
        raw_path = Path("data/raw/criminal_law_npc_test.html")
    if not raw_path.exists():
        print(law["id"], "SKIP no raw")
        continue
    raw = raw_path.read_text(encoding="utf-8")
    arts = parse_law(law, raw)
    print(f"{law['id']}: {len(arts)} articles")
