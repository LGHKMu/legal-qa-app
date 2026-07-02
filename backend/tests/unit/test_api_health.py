"""FastAPI 健康检查与 SSE 契约（无需 API Key / 索引）。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from main import _sse, app


def test_health_and_ready() -> None:
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert "rag_ready" in body

    ready = client.get("/api/ready")
    assert ready.status_code == 200
    assert "ready" in ready.json()


def test_laws_endpoint() -> None:
    client = TestClient(app)
    res = client.get("/api/laws")
    assert res.status_code == 200
    laws = res.json()
    assert isinstance(laws, list)
    if laws:
        assert {"id", "name", "article_count"}.issubset(laws[0].keys())


def test_sse_done_includes_citation_verify_fields() -> None:
    payload = {
        "disclaimer": "免责声明",
        "is_legal": True,
        "citation_verified": True,
        "citation_verify": {
            "passed": True,
            "action": "pass",
            "citation_verified": True,
            "cited_count": 1,
            "invalid_count": 0,
            "invalid": [],
            "warnings": [],
        },
    }
    raw = _sse("done", payload)
    assert raw.startswith("event: done\n")
    data_line = next(line for line in raw.splitlines() if line.startswith("data:"))
    parsed = json.loads(data_line[5:].strip())
    assert parsed["citation_verified"] is True
    assert parsed["citation_verify"]["action"] == "pass"


def test_ask_without_api_key_returns_500() -> None:
    from config import settings

    original = settings.deepseek_api_key
    original_sec = settings.security_require_api_key
    settings.deepseek_api_key = ""
    settings.security_require_api_key = False
    try:
        client = TestClient(app)
        res = client.post("/api/ask", json={"question": "宪法第三十四条是什么？"})
        assert res.status_code == 503
    finally:
        settings.deepseek_api_key = original
        settings.security_require_api_key = original_sec
