import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from context import filter_relevant_history
from classifier import is_legal_question
from config import DISCLAIMER, cors_origins_list, settings
from llm import stream_llm, stream_llm_general
from rag import answer_question, is_ready, list_laws, prepare_answer, warmup
from trace import TraceRecorder
from verify.repair import verify_and_repair


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


def _sse(event: str, data: dict | list | str | bool) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _history_payload(history: list[HistoryMessage]) -> list[dict]:
    return [{"role": h.role, "content": h.content} for h in history]


def _stream_answer(question: str, history: list[HistoryMessage], request_id: str):
    hist = _history_payload(history)
    trace = TraceRecorder(request_id, "/api/ask/stream", question, len(hist))
    answer_parts: list[str] = []
    is_legal: bool | None = None

    yield ": connected\n\n"
    yield _sse("start", {"status": "classifying"})
    try:
        t0 = time.perf_counter()
        relevant = filter_relevant_history(question, hist)
        trace.step(
            "context_filter",
            (time.perf_counter() - t0) * 1000,
            {"context_turns": len(relevant)},
        )

        t0 = time.perf_counter()
        legal = is_legal_question(question, relevant or None)
        is_legal = legal
        trace.step("classify", (time.perf_counter() - t0) * 1000, {"is_legal": legal})
        yield _sse(
            "meta",
            {"is_legal": legal, "context_turns": len(relevant), "request_id": request_id},
        )

        citations: list[dict] = []
        chunks: list[dict] = []
        citation_verified = True
        if legal:
            yield _sse("start", {"status": "retrieving"})
            chunks, citations, _ = prepare_answer(question, history=hist, trace=trace)
            token_stream = stream_llm(question, chunks, relevant)
        else:
            token_stream = stream_llm_general(question, relevant)

        t0 = time.perf_counter()
        for token in token_stream:
            answer_parts.append(token)
            yield _sse("token", {"content": token})
        trace.step(
            "generate",
            (time.perf_counter() - t0) * 1000,
            {"answer_chars": sum(len(t) for t in answer_parts)},
        )

        answer_text = "".join(answer_parts)
        repair = None
        if legal and chunks:
            yield _sse("start", {"status": "verifying"})
            repair = verify_and_repair(
                answer_text,
                chunks,
                question=question,
                history=relevant,
                trace=trace,
            )
            citation_verified = repair.citation_verified
            if repair.answer != answer_text:
                answer_text = repair.answer
                yield _sse(
                    "answer_revision",
                    {"content": answer_text, "action": repair.action},
                )

        if legal and citations:
            yield _sse("citations", citations)

        done_payload: dict = {
            "disclaimer": DISCLAIMER,
            "is_legal": legal,
            "citation_verified": citation_verified,
        }
        if repair is not None:
            verify_data = repair.verify.to_trace_output()
            verify_data["action"] = repair.action
            verify_data["citation_verified"] = repair.citation_verified
            done_payload["citation_verify"] = verify_data

        yield _sse("done", done_payload)
        trace.finish(status="ok", is_legal=is_legal, answer_preview=answer_text)
    except Exception as exc:
        trace.finish(status="error", is_legal=is_legal, error=str(exc))
        yield _sse("error", {"message": str(exc)})


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
    is_legal: bool | None = None
    try:
        result = answer_question(question, history=hist, trace=trace)
        is_legal = result.get("is_legal")
        trace.finish(
            status="ok",
            is_legal=is_legal,
            answer_preview=result.get("answer", ""),
        )
        return result
    except Exception as exc:
        trace.finish(status="error", is_legal=is_legal, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/ask/stream")
def ask_stream(body: AskRequest):
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY")
    request_id = uuid.uuid4().hex[:12]
    return StreamingResponse(
        _stream_answer(body.question.strip(), body.history, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
        },
    )
