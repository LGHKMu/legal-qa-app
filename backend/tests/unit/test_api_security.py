"""API 安全：鉴权、限流、脱敏。"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app
from security import reset_rate_limit_state


@pytest.fixture(autouse=True)
def _clear_rate_limits() -> None:
    reset_rate_limit_state()
    yield
    reset_rate_limit_state()


def test_ask_requires_api_key_when_secured() -> None:
    from config import settings

    original_key = settings.app_api_key
    original_req = settings.security_require_api_key
    original_ds = settings.deepseek_api_key
    settings.app_api_key = "test-secret-key"
    settings.security_require_api_key = True
    settings.deepseek_api_key = "ds-key"
    try:
        client = TestClient(app)
        res = client.post("/api/ask", json={"question": "宪法第三十四条是什么？"})
        assert res.status_code == 401
        ok = client.post(
            "/api/ask",
            json={"question": "宪法第三十四条是什么？"},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert ok.status_code != 401
    finally:
        settings.app_api_key = original_key
        settings.security_require_api_key = original_req
        settings.deepseek_api_key = original_ds


def test_health_public_without_model_by_default() -> None:
    from config import settings

    original = settings.security_expose_model_in_health
    settings.security_expose_model_in_health = False
    try:
        client = TestClient(app)
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert "model" not in body
    finally:
        settings.security_expose_model_in_health = original


def test_rate_limit_returns_429() -> None:
    from config import settings

    original_key = settings.app_api_key
    original_req = settings.security_require_api_key
    original_limit = settings.rate_limit_ask_per_minute
    original_ds = settings.deepseek_api_key
    settings.app_api_key = "k"
    settings.security_require_api_key = True
    settings.rate_limit_ask_per_minute = 2
    settings.deepseek_api_key = "ds"
    headers = {"X-API-Key": "k"}

    with patch("main.run_agent_answer", return_value={"answer": "x", "citations": []}):
        try:
            client = TestClient(app)
            payload = {"question": "测试问题"}
            assert client.post("/api/ask", json=payload, headers=headers).status_code != 429
            assert client.post("/api/ask", json=payload, headers=headers).status_code != 429
            third = client.post("/api/ask", json=payload, headers=headers)
            assert third.status_code == 429
        finally:
            settings.app_api_key = original_key
            settings.security_require_api_key = original_req
            settings.rate_limit_ask_per_minute = original_limit
            settings.deepseek_api_key = original_ds


def test_sanitize_error_hides_internals() -> None:
    from config import settings

    original_sanitize = settings.security_sanitize_errors
    original_key = settings.app_api_key
    original_req = settings.security_require_api_key
    original_ds = settings.deepseek_api_key
    settings.security_sanitize_errors = True
    settings.app_api_key = "k"
    settings.security_require_api_key = True
    settings.deepseek_api_key = "ds"
    headers = {"X-API-Key": "k"}

    with patch("main.run_agent_answer", side_effect=RuntimeError("secret internal path")):
        try:
            client = TestClient(app)
            res = client.post(
                "/api/ask",
                json={"question": "测试"},
                headers=headers,
            )
            assert res.status_code == 500
            assert "secret internal" not in res.json()["detail"]
        finally:
            settings.security_sanitize_errors = original_sanitize
            settings.app_api_key = original_key
            settings.security_require_api_key = original_req
            settings.deepseek_api_key = original_ds
