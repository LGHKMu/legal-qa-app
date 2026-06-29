# 法律智能问答系统

基于 **DeepSeek API** + **RAG（混合检索）** 的法律问答 Web 应用。知识库涵盖《宪法》《民法典》《刑法》《劳动法》（约 1957 条），数据来源于 [国家法律法规数据库](https://flk.npc.gov.cn/) 等官方渠道。

## 功能

- 多轮法律问答（SSE 流式输出 + HTTP POST）
- Cascade 混合检索：向量 + BM25 + 改写 + Cross-Encoder 精排
- 按「条」切分法条，保留编/章/节标题
- 回答强制引用具体法律名称与条号，并做引用校验
- 展示检索到的引用法条、校验状态与免责声明
- 请求 Trace（JSONL 落盘，便于排查与回放）

## 目录结构

```
legal-qa-app/
├── backend/          # FastAPI + ChromaDB + RAG
└── frontend/         # Vue 3 单页
```

## 快速开始

### 1. 后端

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env   # 填入 DEEPSEEK_API_KEY
python scripts/build_index.py
python -m uvicorn main:app --reload --port 8001
```

### 2. 前端

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 http://localhost:5174

### 3. 更新法条数据（可选）

```bash
cd backend
python scripts/fetch_laws.py      # 从官方法律数据库抓取最新原文
python scripts/build_index.py     # 重新向量化入库
```

## Docker 部署（生产 / 演示）

```powershell
# 1. 配置 backend\.env（含 DEEPSEEK_API_KEY）
# 2. 宿主机准备索引与模型（首次）：
#    cd backend
#    python scripts/build_index.py
#    python scripts/download_reranker.py
# 3. 启动
docker compose up -d --build
# 或： .\deploy.ps1
```

浏览器访问 **http://localhost:8080**。健康检查：`/api/health`、`/api/ready`（RAG 预热完成后为 `true`）。

详见 [docs/DEPLOY.md](docs/DEPLOY.md)。

## CI 测试

```powershell
.\run_ci.ps1
```

## API

**POST /api/ask**

```json
{
  "question": "公民的基本权利有哪些？",
  "law_filter": "constitution"
}
```

`law_filter`: `null` | `"constitution"` | `"civil_code"`

## 技术栈

| 组件 | 方案 |
|------|------|
| LLM | DeepSeek `deepseek-chat` |
| Embedding | `BAAI/bge-small-zh-v1.5`（本地） |
| 向量库 | ChromaDB |
| 后端 | FastAPI |
| 前端 | Vue 3 + Element Plus |

## 说明

- 官方法律数据库页面为 SPA，抓取脚本优先调用 `flk.npc.gov.cn/api/detail`；若网络受限，会使用 `data/raw/` 中已缓存的原文。
- 向量索引、BM25、模型权重不随 Git 提交，需在宿主机生成后由 Docker volume 挂载进容器。
- Docker 默认 CPU 推理；有 NVIDIA GPU 时可使用 `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`。

## 免责声明

本系统由 AI 生成，仅供参考，不构成正式法律意见。具体案件请咨询执业律师。
