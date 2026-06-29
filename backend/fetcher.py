from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader

from config import LAWS_YAML, RAW_DIR
from parser import clean_pdf_text, parse_html, parse_plain_text

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://flk.npc.gov.cn/",
    "Origin": "https://flk.npc.gov.cn",
}

NPC_HEADERS = {
    **HEADERS,
    "Referer": "http://www.npc.gov.cn/",
    "Origin": "http://www.npc.gov.cn",
}


def load_laws_config() -> list[dict]:
    with open(LAWS_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [law for law in data.get("laws", []) if law.get("enabled", True)]


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return clean_pdf_text(text)


def _load_local_pdf(law: dict) -> str | None:
    local_pdf = law.get("local_pdf")
    if not local_pdf:
        return None
    src = Path(local_pdf).expanduser()
    if not src.is_file():
        return None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / law["raw_file"]
    if dest.suffix.lower() != ".pdf":
        dest = dest.with_suffix(".pdf")
    shutil.copy2(src, dest)
    return _read_pdf(dest)


def _fetch_flk_html(flk_id: str) -> str | None:
    """从国家法律法规数据库 API 获取 HTML 正文。"""
    try:
        with httpx.Client(follow_redirects=True, timeout=60, headers=HEADERS) as client:
            client.get("https://flk.npc.gov.cn/")
            resp = client.post(
                "https://flk.npc.gov.cn/api/detail",
                data={"id": flk_id},
                headers={**HEADERS, "X-Requested-With": "XMLHttpRequest"},
            )
            if resp.status_code != 200 or not resp.text.strip().startswith("{"):
                return None
            body = resp.json().get("result", {}).get("body", [])
            html_path = next(
                (item.get("url") for item in body if item.get("url")),
                None,
            )
            if not html_path:
                return None
            html_resp = client.get(f"https://wb.flk.npc.gov.cn{html_path}")
            if html_resp.status_code == 200:
                return html_resp.text
    except Exception:
        return None
    return None


def _fetch_url(url: str) -> str | None:
    headers = NPC_HEADERS if "npc.gov.cn" in url else HEADERS
    try:
        resp = httpx.get(url, headers=headers, timeout=60, follow_redirects=True)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        return None
    return None


def _looks_like_law_html(html: str) -> bool:
    return "第" in html and "条" in html and len(html) > 2000


def _looks_like_law_text(text: str) -> bool:
    return "第" in text and "条" in text and len(text) > 2000


def _read_cached_raw(law: dict, raw_path: Path) -> str | None:
    """优先使用本地已缓存的原文。"""
    if not raw_path.exists():
        return None
    if raw_path.suffix.lower() == ".pdf":
        return _read_pdf(raw_path)
    cached = raw_path.read_text(encoding="utf-8")
    if raw_path.suffix.lower() == ".html" and not _looks_like_law_html(cached):
        return None
    if raw_path.suffix.lower() == ".txt" and not _looks_like_law_text(cached):
        return None
    return cached


def fetch_law_raw(law: dict) -> str:
    """抓取法律原文并缓存到 data/raw/。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_name = law["raw_file"]
    raw_path = RAW_DIR / raw_name
    source_url = law.get("source_url", "")

    cached = _read_cached_raw(law, raw_path)
    if cached:
        return cached

    pdf_text = _load_local_pdf(law) if law["raw_file"].endswith(".pdf") else None
    if pdf_text:
        txt_path = raw_path.with_suffix(".txt")
        txt_path.write_text(pdf_text, encoding="utf-8")
        return pdf_text

    def _try_urls(urls: list[str]) -> str | None:
        for url in urls:
            if not url:
                continue
            html = _fetch_url(url)
            if html and _looks_like_law_html(html):
                raw_path.write_text(html, encoding="utf-8")
                return html
        return None

    if not law.get("flk_id") and source_url:
        html = _try_urls([source_url])
        if html:
            return html

    html = _fetch_flk_html(law.get("flk_id", ""))
    if html and _looks_like_law_html(html):
        raw_path.write_text(html, encoding="utf-8")
        return html

    html = _try_urls([source_url, law.get("fallback_url", "")])
    if html:
        return html

    cached_pdf = raw_path if raw_path.suffix.lower() == ".pdf" else raw_path.with_suffix(".pdf")
    if cached_pdf.exists():
        return _read_pdf(cached_pdf)

    txt_path = raw_path.with_suffix(".txt")
    if txt_path.exists() and txt_path != raw_path:
        text = txt_path.read_text(encoding="utf-8")
        if _looks_like_law_text(text):
            return text

    raise FileNotFoundError(
        f"无法获取《{law['name']}》原文，请检查 local_pdf、网络或 {RAW_DIR}"
    )


def parse_law(law: dict, raw: str) -> list:
    source_url = law["source_url"]
    if law["raw_file"].endswith(".html"):
        return parse_html(
            raw,
            law_id=law["id"],
            law_name=law["name"],
            source_url=source_url,
        )
    if law["raw_file"].endswith(".pdf"):
        raw = clean_pdf_text(raw)
    return parse_plain_text(
        raw,
        law_id=law["id"],
        law_name=law["name"],
        source_url=source_url,
    )
