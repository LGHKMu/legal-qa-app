import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.orchestrator import run_agent_answer, stream_agent_answer
from config import DISCLAIMER, cors_origins_list, settings
from rag import is_ready, list_laws, warmup
from trace import TraceRecorder


@asynccontextmanager
async def lifespan(_: FastAPI):
    warmup()
    yield


app = FastAPI(title="法律智能问答", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    history: list[HistoryMessage] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    disclaimer: str = DISCLAIMER
    is_legal: bool = True
    citation_verified: bool = True
    intent: str | None = None
    agent_plan: str | None = None


def _sse(event: str, data: dict | list | str | bool) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _history_payload(history: list[HistoryMessage]) -> list[dict]:
    return [{"role": h.role, "content": h.content} for h in history]


@app.get("/api/health")
def health():
    return {"status": "ok", "model": settings.deepseek_model, "rag_ready": is_ready()}


@app.get("/api/ready")
def ready():
    return {"ready": is_ready()}


@app.get("/api/laws")
def laws():
    return list_laws()


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest):
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY")
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
        trace.finish(status="error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/ask/stream")
def ask_stream(body: AskRequest):
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY")
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
