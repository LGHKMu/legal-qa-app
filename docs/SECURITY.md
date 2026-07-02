# 部署与安全

## 安全模型（三层）

```
浏览器
  ↓  (可选) HTTP Basic Auth — 整站访问密码
nginx :8080
  ↓  limit_req 限流 + 安全响应头
  ↓  注入 X-API-Key（浏览器不可见，仅容器内配置）
backend :8001（不对外暴露端口）
  ↓  API Key 校验 + 应用层限流 + 输入校验 + 错误脱敏
DeepSeek / 本地 RAG
```

| 层级 | 防护 | 配置 |
|------|------|------|
| **入口 nginx** | 限流、Body 64KB、安全头、可选 Basic Auth | `frontend/nginx.conf.template` |
| **后端 FastAPI** | API Key、按 IP 限流、history 截断、错误脱敏 | `SECURITY_*`、`APP_API_KEY` |
| **网络** | 仅暴露 8080，8001 只在 Docker 内网 | `docker-compose.yml` |

---

## 必做（部署前）

### 1. 生成密钥

```powershell
# 项目根 .env
python -c "import secrets; print('APP_API_KEY=' + secrets.token_urlsafe(32))"
```

写入 **项目根** `.env` 的 `APP_API_KEY`（与 `backend/.env` 的 `DEEPSEEK_API_KEY` 不同）。

### 2. 锁定 CORS

`backend/.env` 或 compose 环境变量：

```
CORS_ORIGINS=https://你的域名
```

勿在生产使用 `*`。

### 3. 确认 compose 安全默认

`docker-compose.yml` 已默认：

- `SECURITY_REQUIRE_API_KEY=true`
- `SECURITY_SANITIZE_ERRORS=true`（500 不返回 Python 堆栈）
- backend **无** `ports:`，仅 `expose: 8001`

---

## 可选：HTTP Basic Auth（对外暴露整站时建议）

1. 安装 `htpasswd`（Git for Windows 自带，或 Apache 工具包）

```powershell
cd legal-qa-app\deploy
htpasswd -cb htpasswd admin 你的强密码
```

2. 项目根 `.env`：

```
NGINX_BASIC_AUTH=on
```

3. `docker-compose.yml` 的 frontend 下取消注释：

```yaml
volumes:
  - ./deploy/htpasswd:/etc/nginx/htpasswd:ro
```

4. 重建 frontend：`docker compose up -d --build frontend`

访问站点时会先弹出浏览器用户名/密码框。

---

## 限流说明

| 位置 | 规则 |
|------|------|
| nginx | `/api/ask*` ≈ 10 次/分钟/IP，burst 3 |
| nginx | 其他 `/api/*` ≈ 60 次/分钟/IP |
| FastAPI | `RATE_LIMIT_ASK_PER_MINUTE`（默认 20/分钟/IP） |

两层叠加：即使绕过 nginx 直连内网 backend，后端仍有限流（正常部署无法直连 backend）。

---

## 接口鉴权

| 路径 | API Key | 说明 |
|------|---------|------|
| `GET /api/health` | 否 | 健康检查 |
| `GET /api/ready` | 否 | RAG 就绪 |
| `GET /api/laws` | 否 | 法律列表（只读元数据） |
| `POST /api/ask` | **是** | 问答 |
| `POST /api/ask/stream` | **是** | 流式问答 |

Docker 部署下，Key 由 **nginx 自动注入** `X-API-Key`，前端 SPA 无需改代码。

本地开发（无 Key）在 `backend/.env` 设：

```
SECURITY_REQUIRE_API_KEY=false
```

---

## 输入限制

- 问题：1–500 字
- 历史：最多 20 轮，每条 content ≤ 2000 字
- 请求体：nginx `client_max_body_size 64k`

---

## HTTPS

本地 Docker 默认 **HTTP :8080**（`http://localhost:8080`）。若需 HTTPS，请自行在前置反向代理上配置证书，并更新 `CORS_ORIGINS`。

---

## 检查清单

- [ ] `APP_API_KEY` 已设且未提交 Git
- [ ] `DEEPSEEK_API_KEY` 仅在 `backend/.env`
- [ ] `CORS_ORIGINS` 为实际前端地址（本地一般为 `http://localhost:8080`）
- [ ] 对外暴露时已开 Basic Auth 或等价网关鉴权
- [ ] 未将 backend:8001 映射到宿主机
- [ ] `data/traces` 权限合理（含用户问题文本）

---

## 相关文件

- `backend/security.py` — API Key、限流、脱敏
- `backend/middleware/security_headers.py` — 响应头
- `frontend/nginx.conf.template` — 入口限流与反代
- `docs/DEPLOY.md` — 部署步骤
