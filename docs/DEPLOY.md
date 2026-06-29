# 部署与运维

本文说明如何使用 Docker 部署 **legal-qa-app**（生产/演示环境）。

## 架构

```
浏览器 → frontend (nginx:80) → /api/* 反代 → backend (uvicorn:8001)
                └─ /*           静态文件 dist/
```

- 对外只暴露 **8080**（可改 `APP_PORT`）
- 向量库、BM25、模型权重、Trace 通过 **volume 挂载**，不打进镜像

---

## 前置条件

| 工具 | 用途 |
|------|------|
| Docker Desktop / Docker Engine | 运行容器 |
| Docker Compose v2 | 编排 |
| （可选）NVIDIA Container Toolkit | GPU 推理 |

---

## 第一次部署（推荐流程）

### 1. 配置密钥

```powershell
cd legal-qa-app\backend
copy .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 2. 准备数据（在宿主机执行一次）

索引与模型 **不随 Git 提交**，需在宿主机生成后由 compose 挂载：

```powershell
cd backend
pip install -r requirements.txt
python scripts/build_index.py          # 生成 data/chroma、data/bm25
python scripts/download_reranker.py    # 下载 data/models/bge-reranker-base（若启用 rerank）
```

确认存在：

- `backend/data/chroma/`
- `backend/data/bm25/`
- `backend/data/models/bge-reranker-base/`（若 `RERANK_ENABLED=true`）

### 3. 启动

```powershell
cd legal-qa-app
copy .env.example .env    # 可选：调整 APP_PORT、INFERENCE_DEVICE
docker compose up -d --build
```

浏览器访问：**http://localhost:8080**

### 4. 健康检查

```powershell
curl http://localhost:8080/api/health
curl http://localhost:8080/api/ready
docker compose ps
docker compose logs -f backend
```

`/api/ready` 在 embedding 预热完成后返回 `{"ready": true}`，首次启动可能需要 1–3 分钟。

---

## 常用命令

```powershell
# 启动
docker compose up -d --build

# 停止
docker compose down

# 查看日志
docker compose logs -f backend
docker compose logs -f frontend

# 重建索引（在容器内）
docker compose run --rm backend python scripts/build_index.py

# 仅重启后端
docker compose restart backend
```

---

## GPU 部署（可选）

宿主机已安装 NVIDIA 驱动与 [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 时：

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

并在 `backend/.env` 中设置 `INFERENCE_DEVICE=cuda`（gpu compose 文件会覆盖为 cuda）。

---

## 环境变量

### 项目根 `.env`（compose 读取）

| 变量 | 默认 | 说明 |
|------|------|------|
| `APP_PORT` | 8080 | 对外端口 |
| `INFERENCE_DEVICE` | cpu | 容器内推理设备 |
| `CORS_ORIGINS` | http://localhost:8080 | 后端 CORS 白名单 |

### `backend/.env`（应用配置）

见 `backend/.env.example`。Docker 部署时至少配置：

- `DEEPSEEK_API_KEY`（必填）
- `CORS_ORIGINS`（与前端访问域名一致）

---

## 卷挂载说明

| 宿主机路径 | 容器路径 | 说明 |
|------------|----------|------|
| `backend/data/chroma` | `/app/data/chroma` | 向量索引 |
| `backend/data/bm25` | `/app/data/bm25` | BM25 索引 |
| `backend/data/models` | `/app/data/models` | embedding / reranker 权重 |
| `backend/data/traces` | `/app/data/traces` | 请求 Trace JSONL |

更新法条后：宿主机或容器内重新 `build_index.py`，然后 `docker compose restart backend`。

---

## 与本地开发的区别

| | 本地 dev | Docker |
|--|----------|--------|
| 前端 | `npm run dev` :5174 | nginx :8080 |
| 后端 | uvicorn :8001 | 容器内 8001，不直接暴露 |
| GPU | 本机 CUDA | 默认 CPU；可选 gpu compose |
| 依赖 | 本机 Python/Node | 镜像内已安装 |

本地开发仍用：

```powershell
cd backend && python -m uvicorn main:app --port 8001
cd frontend && npm run dev
```

---

## 故障排查

| 现象 | 处理 |
|------|------|
| backend 启动失败 `DEEPSEEK_API_KEY` | 检查 `backend/.env` |
| `/api/ready` 长期 false | 确认 `data/chroma` 已挂载且非空；看 `docker compose logs backend` |
| 首次问答很慢 | embedding/reranker 首次加载；CPU 更慢属正常 |
| SSE 中断 | nginx 已关闭 buffering；确认经 **8080** 访问而非直连 8001 |
| 构建镜像慢 | sentence-transformers / torch 体积大，首次 build 约 10–20 分钟 |

---

## 生产建议（简要）

1. `CORS_ORIGINS` 设为实际域名，勿用 `*`
2. 前置 HTTPS 反向代理（Caddy / Nginx / 云 LB）
3. 定期备份 `data/chroma`、`data/parsed`
4. 后续可叠加 API Key 中间件、限流（见路线图）

---

## 一键脚本（Windows）

```powershell
# 项目根目录
.\deploy.ps1        # 检查 .env 与索引目录后 compose up
.\deploy.ps1 -Down  # 停止
```
