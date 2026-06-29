"""请求级 Trace：一次 HTTP 请求一条 JSONL 记录。"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from config import settings

_lock = threading.Lock()


class TraceRecorder:
    def __init__(
        self,
        request_id: str,
        endpoint: str,
        question: str,
        history_turns: int,
    ):
        self._request_id = request_id
        self._endpoint = endpoint
        self._question = question
        self._history_turns = history_turns
        self._steps: list[dict] = []
        self._started = time.perf_counter()

    def step(self, name: str, ms: float, output: dict | None = None) -> None:
        entry: dict = {"name": name, "ms": round(ms, 1)}
        if output:
            entry["output"] = output
        self._steps.append(entry)

    def finish(
        self,
        *,
        status: str,
        is_legal: bool | None = None,
        answer_preview: str = "",
        error: str | None = None,
    ) -> None:
        if not settings.trace_enabled:
            return
        record = {
            "request_id": self._request_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "endpoint": self._endpoint,
            "status": status,
            "latency_ms": round((time.perf_counter() - self._started) * 1000, 1),
            "question": self._question,
            "history_turns": self._history_turns,
            "is_legal": is_legal,
            "answer_preview": answer_preview[:200],
            "steps": self._steps,
            "error": error,
        }
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = Path(settings.trace_dir) / f"{day}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _lock:
            path.open("a", encoding="utf-8").write(line)
