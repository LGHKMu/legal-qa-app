# Docker 冒烟测试清单

部署完成后按此清单验收（约 5 分钟）。

## 1. 启动

```powershell
cd legal-qa-app
.\deploy.ps1
# 或： docker compose up -d --build
```

## 2. 容器状态

```powershell
docker compose ps
```

| 容器 | 期望 |
|------|------|
| `legal-qa-backend` | `healthy` |
| `legal-qa-frontend` | `Up` |

## 3. API 检查

```powershell
curl http://localhost:8080/api/health
curl http://localhost:8080/api/ready
curl http://localhost:8080/api/laws
```

| 端点 | 期望 |
|------|------|
| `/api/health` | `rag_ready: true`（预热完成后） |
| `/api/ready` | `{"ready": true}` |
| `/api/laws` | 4 部法律 JSON 列表 |

## 4. 问答测试

浏览器打开 **http://localhost:8080**，输入：

> 劳动合同试用期最长多久？

检查：

- [ ] 回答流式输出
- [ ] 下方展示引用法条
- [ ] 显示引用校验状态（已校验 / 未通过等）

或使用 API：

```powershell
curl -X POST http://localhost:8080/api/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"劳动合同试用期最长多久？\"}"
```

## 5. Trace 落盘（可选）

问答后在宿主机查看：

```powershell
Get-Content backend\data\traces\*.jsonl -Tail 1
```

应包含 `request_id`、`steps`、`status: ok`。

## 6. 停止

```powershell
.\deploy.ps1 -Down
# 或： docker compose down
```

## 常见问题

见 [DEPLOY.md](./DEPLOY.md#故障排查)。
