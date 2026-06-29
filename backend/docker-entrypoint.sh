#!/bin/sh
set -e

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "[ERROR] 未设置 DEEPSEEK_API_KEY。请在 backend/.env 或 docker compose 环境中配置。"
  exit 1
fi

HOST="${UVICORN_HOST:-0.0.0.0}"
PORT="${UVICORN_PORT:-8001}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "[backend] 启动 uvicorn ${HOST}:${PORT} (log=${LOG_LEVEL})"
exec uvicorn main:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
