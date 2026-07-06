# 法律智能问答系统

基于 **DeepSeek API** + **RAG（混合检索）** 的法律问答 Web 应用。知识库涵盖《宪法》《民法典》《刑法》《劳动法》（约 1957 条），数据来源于 [国家法律法规数据库](https://flk.npc.gov.cn/) 等官方渠道。

## 功能

- 多轮法律问答（SSE 流式输出 + HTTP POST）
- **Agent 编排**：按问题类型（查条 / 概念 / 案情 / 非法律）自动选择执行计划，流式展示计划与步骤进度
- Cascade 混合检索：向量 + BM25 + 改写 + Cross-Encoder 精排
- **案情咨询补充检索**：首轮改写检索质量不足时，自动二轮补充并合并结果（仅案情类问题）
- **主题检索增强**：对「高空抛物」等易歧义主题，自动加消歧词并保底核心法条（如民法典第 1254、1188 条）
- 按「条」切分法条，保留编/章/节标题
- 回答强制引用具体法律名称与条号，并做引用校验（【法律依据】与引用卡片对齐）
- 展示检索到的引用法条、校验状态与免责声明
- 请求 Trace（JSONL 落盘，便于排查与回放）

## 近期更新（Agent 与检索增强）

### Agent 问答（Phase 1）

请求进入后先 **意图路由**（规则优先，必要时 LLM 兜底），再按类型执行固定工具链：

| 意图 | 说明 | 典型问法 |
|------|------|----------|
| `statute_lookup` | 法条查询 | 民法典第 1046 条是什么 |
| `concept_qa` | 概念解释 | 公民的基本权利有哪些 |
| `case_consult` | 案情咨询 | 加班不给钱怎么办 |
| `non_legal` | 非法律闲聊 | 今天天气怎么样 |

流式接口新增 SSE 事件：`agent_plan`（计划与意图）、`agent_step`（步骤状态/耗时）、`agent_retry`（案情补充检索提示）。

### 引用校验对齐

- 校验后【法律依据】仅保留本次检索/查条结果中的法条，与前端「引用法条」卡片一致
- 查条、概念、案情使用不同生成 prompt，减少臆造条号
- 校验修正回答时，通过 `answer_revision` 事件更新正文

### 案情检索 Retry（Phase 2）

仅 **案情咨询** 触发，Agent 计划步数不变（retry 内嵌在「检索相关法条」工具内）：

1. 首轮：`rewrite=True` 混合检索
2. 规则评估：top 分、分差、主题是否匹配等
3. 不足则二轮：`rewrite=False` 补充检索（不调 LLM 改写），与首轮 merge
4. 命中主题规则时，二轮使用规则增强 query（如加「抛掷物品」），避免原问中「高空」等歧义词

相关配置见 `backend/.env.example` 中 `AGENT_CASE_RETRY_*` 项。

### 主题检索规则

在 `query_rewrite.py` 中维护主题表（非单题 hardcode）：

- **Query 增强**：改写后叠加消歧关键词
- **主题相关性检测**：结果与问题主题不符则触发补充检索
- **锚点保底**：关键法条缺失时从 parsed JSON 精确注入

扩展新主题时，在 `TOPIC_SEARCH_RULES` / `TOPIC_ANCHOR_ARTICLES` 增加配置即可。

### 检索性能优化（P0）

| 能力 | 说明 | 配置 |
|------|------|------|
| **查条短路** | `statute_lookup` 且 `get_article` 命中 parsed JSON 时，跳过混合检索 | 无需配置 |
| **法律域硬过滤** | 高置信度单一 `law_id` 时传给 `retrieve_fusion`（Chroma `where` + BM25 过滤） | `AGENT_LAW_FILTER_ENABLED`、`AGENT_LAW_FILTER_MIN_CONFIDENCE` |
| **查条 fallback 无改写** | 查条未命中 JSON 时的 fallback 检索不走 LLM 改写 | 自动（`statute_lookup`） |
| **fast 档** | 关 concat/union、缩小候选池，降低 CPU 精排与 BM25 开销 | `RAG_PROFILE=fast` |

演示或低延迟部署可设 `RAG_PROFILE=fast`；默认 `accurate` 保持评测 Recall 口径不变。

### Agent 检索增强（P1）

- **案情首轮规则 query**：无 LLM 改写时，命中主题规则（如高空抛物）首轮即用规则增强 query，不再等二轮 retry
- **Retry 降噪**：主题已匹配且 top 分够高时，不因「条与条分数接近」误触发二轮检索
- **查条短路反馈**：`get_article` 命中时 SSE 步骤带 `shortcut` / `skipped_search`，前端可展示「已精确查条」

### 评测口径（与线上一致）

| 列 / 命令 | 含义 | 与 `/api/ask` 关系 |
|-----------|------|-------------------|
| **混合列** | 全题 `retrieve_fusion(rewrite=True)` | 检索消融对照，**非**线上默认 |
| **Agent 列** | `agent.pre_retrieval.run_agent_pre_retrieval` | **与线上一致**（路由 + 查条/fallback + 案情 retry） |

```bash
cd backend
# 四列对比（含 Agent 线上路径）
python scripts/compare_rag.py --compare-rewrite --retrieval-only

# 仅 Agent 列
python scripts/compare_rag.py --compare-agent --retrieval-only
```

**线上口径以 Agent 列为准**；混合列用于对比 Cascade+Union 检索管线本身的上限。

### CI Recall 门禁

| Profile | 说明 | 阈值 | API Key |
|---------|------|------|---------|
| `ci_no_llm`（默认 CI） | Agent 线上路径，无 LLM 改写 | Recall@5 ≥ 72% | 不需要 |
| `ci_full`（手动 workflow，勾选 run_full_recall_gate） | Agent 含 LLM 改写 | Recall@5 ≥ 85% | 需要 `DEEPSEEK_API_KEY` secret |

```bash
cd backend
python scripts/recall_gate.py                    # 默认 ci_no_llm
python scripts/recall_gate.py --profile ci_full  # 完整线上口径
python scripts/compare_rag.py --compare-agent --retrieval-only --min-recall 0.72 --gate-mode agent
```

阈值配置见 `backend/data/recall_gate.yaml`；GitHub Actions 在 `recall-gate` job 中自动运行。

## 目录结构

```
legal-qa-app/
├── backend/
│   ├── agent/          # Agent 路由、计划、工具、编排
│   ├── verify/         # 引用抽取与校验修复
│   └── ...
└── frontend/           # Vue 3 单页（含 Agent 步骤 UI）
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

## Docker 部署（安全优先）

```powershell
# 1. 根目录 .env：APP_API_KEY（随机密钥，见 docs/SECURITY.md）
# 2. backend\.env：DEEPSEEK_API_KEY、CORS_ORIGINS
# 3. 宿主机准备索引（首次）：cd backend && python scripts/build_index.py
# 4. 启动
copy .env.example .env   # 填入 APP_API_KEY
.\deploy.ps1
```

- 对外仅 **8080**；backend **8001 不暴露**
- 问答需 **APP_API_KEY**（nginx 自动注入，前端无需改代码）
- 对外暴露时可开启 **Basic Auth**，见 [docs/SECURITY.md](docs/SECURITY.md)

浏览器访问 **http://localhost:8080**。健康检查：`/api/health`、`/api/ready`。

详见 [docs/DEPLOY.md](docs/DEPLOY.md)。

## CI 测试

```powershell
.\run_ci.ps1
```

后端单元测试（含 Agent 路由、引用校验、检索质量、主题词规则）：

```bash
cd backend
python -m pytest tests/unit -q
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
