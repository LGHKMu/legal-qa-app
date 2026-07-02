"""应用层安全：API Key、限流、客户端 IP、错误脱敏。"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from config import settings

_RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


def client_ip(request: Request) -> str:
    """从反向代理头解析客户端 IP（需 nginx 设置 X-Real-IP / X-Forwarded-For）。"""
    if settings.security_trust_proxy_headers:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _api_key_enabled() -> bool:
    return bool(settings.security_require_api_key and settings.app_api_key.strip())


def verify_api_key(request: Request) -> None:
    """校验 X-API-Key 或 Authorization: Bearer（部署时由 nginx 注入或客户端携带）。"""
    if not _api_key_enabled():
        return
    expected = settings.app_api_key.strip()
    header_key = (request.headers.get("X-API-Key") or "").strip()
    auth = (request.headers.get("Authorization") or "").strip()
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    provided = header_key or bearer
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未授权访问",
            headers={"WWW-Authenticate": "Bearer"},
        )


def check_rate_limit(request: Request, *, scope: str = "ask") -> None:
    """按 IP 滑动窗口限流（/api/ask*）。"""
    limit = settings.rate_limit_ask_per_minute
    if limit <= 0:
        return
    key = f"{scope}:{client_ip(request)}"
    now = time.time()
    window = 60.0
    bucket = _RATE_BUCKETS[key]
    while bucket and bucket[0] <= now - window:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后再试",
        )
    bucket.append(now)


def sanitize_error_detail(exc: Exception) -> str:
    """生产环境不向客户端返回内部异常字符串。"""
    if settings.security_sanitize_errors:
        return "服务暂时不可用，请稍后重试"
    return str(exc) or "unknown error"


def reset_rate_limit_state() -> None:
    """测试用：清空限流桶。"""
    _RATE_BUCKETS.clear()
