# 部署与运维（安全优先）

**先读 [SECURITY.md](./SECURITY.md)**，完成密钥与 CORS 配置后再启动。

## 架构

```
浏览器 → frontend nginx :8080（限流 / 可选 Basic Auth）
              └─ /api/* → backend :8001（Docker 内网，注入 X-API-Key）
              └─ /*     静态页面
```

- 对外只暴露 **8080**
- backend **8001 不映射到宿主机**
- 索引与模型通过 volume 挂载

---

## 第一次部署

### 1. 密钥与配置

```powershell
cd legal-qa-app

# 项目根：compose 与安全变量
copy .env.example .env
# 编辑 .env：生成并填入 APP_API_KEY

cd backend
copy .env.example .env
# 编辑 .env：填入 DEEPSEEK_API_KEY、CORS_ORIGINS
```

生成 `APP_API_KEY`：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. 准备索引（宿主机一次）

```powershell
cd backend
pip install -r requirements.txt
python scripts/build_index.py
python scripts/download_reranker.py   # 若启用 rerank
```

### 3. 启动

```powershell
cd legal-qa-app
.\deploy.ps1
```

访问：**http://localhost:8080**

### 4. 验证

```powershell
curl http://localhost:8080/api/health
curl http://localhost:8080/api/ready
# 无 Key 应 401：
curl -X POST http://localhost:8080/api/ask -H "Content-Type: application/json" -d "{\"question\":\"test\"}"
```

---

## 常用命令

```powershell
docker compose up -d --build
docker compose down
docker compose logs -f backend
docker compose logs -f frontend
docker compose restart backend
```

---

## GPU（可选）

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

`backend/.env` 设 `INFERENCE_DEVICE=cuda`。

---

## 环境变量

| 位置 | 变量 | 说明 |
|------|------|------|
| 根 `.env` | `APP_API_KEY` | 问答鉴权（必填） |
| 根 `.env` | `APP_PORT` | 对外端口，默认 8080 |
| 根 `.env` | `NGINX_BASIC_AUTH` | `on` 启用整站密码 |
| `backend/.env` | `DEEPSEEK_API_KEY` | LLM（必填） |
| `backend/.env` | `CORS_ORIGINS` | 前端域名白名单 |

---

## 故障排查

| 现象 | 处理 |
|------|------|
| POST /api/ask 401 | 检查根 `.env` `APP_API_KEY` 与 frontend `BACKEND_API_KEY` 一致 |
| POST /api/ask 429 | 触发限流，稍后重试 |
| POST /api/ask 503 | `DEEPSEEK_API_KEY` 或 `APP_API_KEY` 未配置 |
| Basic Auth 启动失败 | 未挂载 `deploy/htpasswd` |
| `/api/ready` false | 检查 `data/chroma` 挂载 |

---

## 本地开发（无 Docker）

```powershell
# backend/.env
SECURITY_REQUIRE_API_KEY=false

cd backend && python -m uvicorn main:app --port 8001
cd frontend && npm run dev
```

开发环境勿使用生产 `APP_API_KEY`。
