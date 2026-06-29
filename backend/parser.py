import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

CN_NUM = r"[零〇一二三四五六七八九十百千万\d]+"
ARTICLE_NO = rf"第{CN_NUM}条(?:之{CN_NUM})?"
ARTICLE_RE = re.compile(ARTICLE_NO)
ARTICLE_SPLIT = re.compile(rf"({ARTICLE_NO})")
LINE_ARTICLE_RE = re.compile(rf"^({ARTICLE_NO})\s*(.*)")
BIAN_RE = re.compile(rf"第{CN_NUM}编\s*[^\n]{{0,40}}")
FENBIAN_RE = re.compile(rf"第{CN_NUM}分编\s*[^\n]{{0,40}}")
CHAPTER_RE = re.compile(rf"第{CN_NUM}章\s*[^\n]{{0,40}}")
SECTION_RE = re.compile(rf"第{CN_NUM}节\s*[^\n]{{0,40}}")


@dataclass
class Article:
    law_id: str
    law_name: str
    article_no: str
    text: str
    hierarchy: str
    source_url: str

    @property
    def doc_id(self) -> str:
        safe = re.sub(r"[^\w]", "_", self.article_no)
        return f"{self.law_id}_{safe}"

    def embed_text(self) -> str:
        parts = [f"《{self.law_name}》"]
        if self.hierarchy:
            parts.append(self.hierarchy)
        parts.append(self.article_no)
        parts.append(self.text)
        return "\n".join(parts)


def _normalize(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ").replace("　", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pdf_text(text: str) -> str:
    """去除 PDF 中常见的页眉页脚噪声。"""
    skip = (
        "English", "无障碍浏览", "适老关怀版", "分享到", "打印",
        "首页", "机构设置", "全面依法治国", "政府信息公开",
    )
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or any(s.startswith(k) for k in skip):
            continue
        if re.fullmatch(r"[-\d/ ]+", s):
            continue
        lines.append(s)
    return _normalize("\n".join(lines))


def _update_hierarchy(line: str, state: dict[str, str]) -> None:
    for key, pattern in (
        ("bian", BIAN_RE),
        ("fenbian", FENBIAN_RE),
        ("chapter", CHAPTER_RE),
        ("section", SECTION_RE),
    ):
        match = pattern.search(line)
        if match:
            state[key] = match.group(0).strip()
            if key == "bian":
                state["fenbian"] = ""
                state["chapter"] = ""
                state["section"] = ""
            elif key == "fenbian":
                state["chapter"] = ""
                state["section"] = ""
            elif key == "chapter":
                state["section"] = ""


def _hierarchy_label(state: dict[str, str]) -> str:
    parts = [state[k] for k in ("bian", "fenbian", "chapter", "section") if state.get(k)]
    return " > ".join(parts)


def _prepare_law_text(text: str) -> str:
    """统一法条标题格式，便于按行首切分。"""
    text = re.sub(rf"(?<=[\s>])({ARTICLE_NO})\s*", r"\n\1 ", text)
    text = re.sub(rf"\n({ARTICLE_NO})\s+", r"\n\1 ", text)
    return text


def parse_plain_text(
    text: str,
    *,
    law_id: str,
    law_name: str,
    source_url: str,
) -> list[Article]:
    """按行首「第X条」切分，避免正文引用（如「依照第121条」）误切。"""
    text = _normalize(text)
    text = _prepare_law_text(text)
    state: dict[str, str] = {}
    articles: list[Article] = []
    current: Article | None = None
    started = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current is not None:
                current.text = f"{current.text}\n" if current.text else ""
            continue

        art_match = LINE_ARTICLE_RE.match(line)
        if art_match:
            started = True
            if current is not None:
                articles.append(current)
            article_no, first_part = art_match.groups()
            current = Article(
                law_id=law_id,
                law_name=law_name,
                article_no=article_no.strip(),
                text=first_part.strip(),
                hierarchy=_hierarchy_label(state),
                source_url=source_url,
            )
            continue

        if not started:
            _update_hierarchy(line, state)
            continue

        _update_hierarchy(line, state)
        if current is not None:
            current.text = f"{current.text}\n{line}" if current.text else line

    if current is not None:
        articles.append(current)

    if articles:
        return articles
    return _parse_plain_text_legacy(text, law_id=law_id, law_name=law_name, source_url=source_url)


def _parse_plain_text_legacy(
    text: str,
    *,
    law_id: str,
    law_name: str,
    source_url: str,
) -> list[Article]:
    """回退：全文正则切分（兼容行首无法识别条号的 PDF 文本）。"""
    state: dict[str, str] = {}
    articles: list[Article] = []

    for line in text.splitlines():
        _update_hierarchy(line, state)

    parts = ARTICLE_SPLIT.split(text)
    i = 1
    while i < len(parts):
        article_no = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        body = re.split(rf"第{CN_NUM}[编分章节]", body)[0].strip()
        if body:
            articles.append(
                Article(
                    law_id=law_id,
                    law_name=law_name,
                    article_no=article_no,
                    text=body,
                    hierarchy=_hierarchy_label(state),
                    source_url=source_url,
                )
            )
        i += 2

    return _attach_hierarchy_by_scan(text, articles, law_id, law_name, source_url)


def _attach_hierarchy_by_scan(
    text: str,
    articles: list[Article],
    law_id: str,
    law_name: str,
    source_url: str,
) -> list[Article]:
    """按文本顺序为每条法条补全章节标题。"""
    if not articles:
        return []

    state: dict[str, str] = {}
    by_no: dict[str, Article] = {a.article_no: a for a in articles}
    result: list[Article] = []

    for segment in ARTICLE_SPLIT.split(text):
        if not segment:
            continue
        if ARTICLE_RE.fullmatch(segment.strip()):
            article = by_no.get(segment.strip())
            if article:
                result.append(
                    Article(
                        law_id=law_id,
                        law_name=law_name,
                        article_no=article.article_no,
                        text=article.text,
                        hierarchy=_hierarchy_label(state),
                        source_url=source_url,
                    )
                )
        else:
            for line in segment.splitlines():
                _update_hierarchy(line, state)

    return result or articles


def dedupe_articles(articles: list[Article]) -> list[Article]:
    """同一法条号出现多次时，保留正文最长的一条。"""
    best: dict[str, Article] = {}
    for article in articles:
        prev = best.get(article.article_no)
        if prev is None or len(article.text) > len(prev.text):
            best[article.article_no] = article
    return list(best.values())


def _extract_html_text(soup: BeautifulSoup) -> str:
    """从 HTML 中选取正文最长的候选节点（避免误选页头 .content 等空容器）。"""
    selectors = (
        "#Zoom",
        ".pages_content",
        "#BodyLabel",
        "#tex",
        ".ejxxgk_xq_con",
        ".TRS_Editor",
        "article",
        ".content",
    )
    candidates: list[str] = []
    for selector in selectors:
        for node in soup.select(selector):
            text = _normalize(node.get_text("\n", strip=True))
            if text:
                candidates.append(text)
    if soup.body:
        body_text = _normalize(soup.body.get_text("\n", strip=True))
        if body_text:
            candidates.append(body_text)
    return max(candidates, key=len) if candidates else ""


def parse_html(
    html: str,
    *,
    law_id: str,
    law_name: str,
    source_url: str,
) -> list[Article]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = _extract_html_text(soup) or _normalize(html)
    text = _prepare_law_text(text)
    return dedupe_articles(
        parse_plain_text(text, law_id=law_id, law_name=law_name, source_url=source_url)
    )
