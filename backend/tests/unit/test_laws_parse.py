"""四部法律解析与 parsed 缓存完整性。"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from config import DATA_DIR, LAWS_YAML
from fetcher import parse_law

MIN_ARTICLES = {
    "constitution": 100,
    "civil_code": 1000,
    "criminal_law": 400,
    "labor_law": 80,
}


def test_parsed_json_exists_and_nonempty() -> None:
    parsed_dir = DATA_DIR / "parsed"
    for law_id, minimum in MIN_ARTICLES.items():
        path = parsed_dir / f"{law_id}.json"
        assert path.is_file(), f"missing {path}"
        data = json.loads(path.read_text(encoding="utf-8"))
        articles = data.get("articles", [])
        assert len(articles) >= minimum, f"{law_id} only {len(articles)} articles"


def test_parse_raw_laws() -> None:
    laws = yaml.safe_load(LAWS_YAML.read_text(encoding="utf-8"))["laws"]
    for law in laws:
        if not law.get("enabled", True):
            continue
        raw_path = Path("data/raw") / law["raw_file"]
        if raw_path.suffix == ".pdf":
            raw_path = raw_path.with_suffix(".txt")
        if not raw_path.exists() and law["id"] == "criminal_law":
            raw_path = Path("data/raw/criminal_law_npc_test.html")
        if not raw_path.exists():
            continue
        raw = raw_path.read_text(encoding="utf-8")
        arts = parse_law(law, raw)
        assert len(arts) >= MIN_ARTICLES[law["id"]], law["id"]
