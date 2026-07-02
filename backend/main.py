import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from agent.orchestrator import run_agent_answer, stream_agent_answer
from config import DISCLAIMER, cors_origins_list, settings
from middleware.security_headers import SecurityHeadersMiddleware
from rag import is_ready, list_laws, warmup
from security import (
    check_rate_limit,
    sanitize_error_detail,
    verify_api_key,
)
from trace import TraceRecorder

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.security_require_api_key and not settings.app_api_key.strip():
        logger.warning(
            "SECURITY_REQUIRE_API_KEY=true 但未设置 APP_API_KEY，问答接口将拒绝访问"
        )
    warmup()
    yield


app = FastAPI(title="法律智能问答", version="1.0.0", lifespan=lifespan)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_list(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-Id"],
)


class HistoryMessage(BaseModel):
    role: str
    content: str = Field(..., max_length=2000)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        role = value.strip().lower()
        if role not in {"user", "assistant"}:
            raise ValueError("role 必须为 user 或 assistant")
        return role


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)

    @field_validator("history")
    @classmethod
    def clamp_history(cls, value: list[HistoryMessage]) -> list[HistoryMessage]:
        max_turns = settings.ask_max_history_turns
        if len(value) > max_turns:
            return value[-max_turns:]
        return value


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    disclaimer: str = DISCLAIMER
    is_legal: bool = True
    citation_verified: bool = True
    intent: str | None = None
    agent_plan: str | None = None


def _secure_ask_deps(request: Request) -> None:
    verify_api_key(request)
    check_rate_limit(request)


def _sse(event: str, data: dict | list | str | bool) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _history_payload(history: list[HistoryMessage]) -> list[dict]:
    max_chars = settings.ask_max_history_content_chars
    return [
        {
            "role": h.role,
            "content": h.content[:max_chars],
        }
        for h in history
    ]


@app.get("/api/health")
def health():
    payload: dict = {"status": "ok", "rag_ready": is_ready()}
    if settings.security_expose_model_in_health:
        payload["model"] = settings.deepseek_model
    return payload


@app.get("/api/ready")
def ready():
    return {"ready": is_ready()}


@app.get("/api/laws")
def laws():
    return list_laws()


@app.post("/api/ask", response_model=AskResponse, dependencies=[Depends(_secure_ask_deps)])
def ask(body: AskRequest):
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=503, detail="服务未就绪")
    if settings.security_require_api_key and not settings.app_api_key.strip():
        raise HTTPException(status_code=503, detail="服务未就绪")
    question = body.question.strip()
    hist = _history_payload(body.history)
    request_id = uuid.uuid4().hex[:12]
    trace = TraceRecorder(request_id, "/api/ask", question, len(hist))
    try:
        result = run_agent_answer(question, history=hist, trace=trace)
        trace.finish(
            status="ok",
            is_legal=result.get("is_legal"),
            answer_preview=result.get("answer", ""),
        )
        return result
    except Exception as exc:
        logger.exception("ask failed request_id=%s", request_id)
        trace.finish(status="error", error=str(exc))
        raise HTTPException(status_code=500, detail=sanitize_error_detail(exc)) from exc


@app.post("/api/ask/stream", dependencies=[Depends(_secure_ask_deps)])
def ask_stream(body: AskRequest):
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=503, detail="服务未就绪")
    if settings.security_require_api_key and not settings.app_api_key.strip():
        raise HTTPException(status_code=503, detail="服务未就绪")
    request_id = uuid.uuid4().hex[:12]
    trace = TraceRecorder(
        request_id,
        "/api/ask/stream",
        body.question.strip(),
        len(body.history),
    )
    return StreamingResponse(
        stream_agent_answer(
            body.question.strip(),
            _history_payload(body.history),
            request_id,
            trace,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
        },
    )
