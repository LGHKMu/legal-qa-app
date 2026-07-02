from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
LAWS_YAML = DATA_DIR / "laws.yaml"
INDEX_STATS_FILE = DATA_DIR / "index_stats.json"
BM25_INDEX_DIR = DATA_DIR / "bm25"
BM25_INDEX_FILE = BM25_INDEX_DIR / "index.pkl"
TRACE_DIR = DATA_DIR / "traces"

DISCLAIMER = (
    "本系统由 AI 生成，仅供参考，不构成正式法律意见。"
    "具体案件请咨询执业律师或司法机关。"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_local_only: bool = True
    chroma_dir: str = str(DATA_DIR / "chroma")
    top_k: int = 5
    retrieve_candidate_k: int = 30
    rrf_k: int = 60
    query_rewrite_enabled: bool = True
    query_rewrite_mode: str = "two_stage"
    query_rewrite_max_tokens: int = 64
    query_extract_max_tokens: int = 256
    # Cross-Encoder：Cascade 池精排 + 改写列 union 精排
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_model_path: str = ""
    rerank_candidate_k: int = 40
    rerank_local_only: bool = True
    rerank_mmr_lambda: float = 0.85
    rerank_mmr_pure_lead: int = 1
    rerank_mmr_diversify_same_law: bool = False
    # BM25 稀疏检索（需 build_index.py 同步构建）
    bm25_enabled: bool = True
    bm25_candidate_k: int = 20
    bm25_rrf_max_entries: int = 5
    bm25_rrf_weight: float = 0.5
    # Cascade 建池：路径保底 + RRF 填满
    rrf_pool_k: int = 40
    path_reserve_vector_top: int = 2
    path_reserve_bm25_top: int = 1
    domain_rrf_boost: float = 1.15
    domain_boost_min_confidence: float = 0.7
    concat_retrieval_enabled: bool = True
    concat_rrf_weight: float = 1.15
    # 精排：案情→改写 query；概念→原问/改写/concat 加权；条号→原问
    rerank_weight_orig: float = 0.2
    rerank_weight_rewrite: float = 0.5
    rerank_weight_concat: float = 0.3
    rerank_query_max_chars: int = 384
    rerank_group_cluster_threshold: float = 0.78
    # 精排后截断：相对首条 α（始终）+ 相邻跌幅 γ（min_k 之后）
    rerank_gap_truncate_enabled: bool = True
    rerank_truncate_min_relative: float = 0.72
    rerank_truncate_max_step_drop: float = 0.25
    rerank_gap_truncate_min: int = 2
    # embedding / reranker：auto | cuda | cuda:0 | cpu
    inference_device: str = "auto"
    # 请求 Trace（JSONL 落盘）
    trace_enabled: bool = True
    trace_dir: str = str(TRACE_DIR)
    # 回答引用校验
    citation_verify_enabled: bool = True
    citation_verify_repair_enabled: bool = True
    # 部署：CORS 白名单（逗号分隔；* 表示允许全部，生产建议指定前端域名）
    cors_origins: str = "*"
    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 8001
    log_level: str = "info"
    # 安全（部署建议开启 REQUIRE + APP_API_KEY，见 docs/SECURITY.md）
    security_require_api_key: bool = False
    app_api_key: str = ""
    rate_limit_ask_per_minute: int = 20
    security_sanitize_errors: bool = True
    security_trust_proxy_headers: bool = True
    security_expose_model_in_health: bool = False
    ask_max_history_turns: int = 20
    ask_max_history_content_chars: int = 2000
    # Agent 运行时
    agent_enabled: bool = True
    agent_router_llm_enabled: bool = True
    # Agent Phase 2：案情咨询检索不足时用原问 baseline 补搜
    agent_case_retry_enabled: bool = True
    agent_case_retry_min_top_score: float = 0.55
    agent_case_retry_min_score_gap: float = 0.08
    agent_case_retry_max_law_ids: int = 2
    agent_case_retry_min_domain_conf: float = 0.7
    # 检索档位：accurate（默认全量 Cascade+Union）| fast（演示/低延迟）
    rag_profile: str = "accurate"
    # Agent 检索：高置信度法律域硬过滤（Chroma where + BM25 范围）
    agent_law_filter_enabled: bool = True
    agent_law_filter_min_confidence: float = 0.7
    # 改写列 ∪ 混合列 的 union 精排（fast 档可关）
    rewrite_union_rerank_enabled: bool = True


settings = Settings()


def apply_rag_profile() -> None:
    """按 RAG_PROFILE 覆盖检索参数（在 settings 加载后调用）。"""
    profile = (settings.rag_profile or "accurate").strip().lower()
    if profile != "fast":
        return
    settings.concat_retrieval_enabled = False
    settings.rewrite_union_rerank_enabled = False
    settings.rrf_pool_k = min(settings.rrf_pool_k, 20)
    settings.rerank_candidate_k = min(settings.rerank_candidate_k, 25)
    settings.retrieve_candidate_k = min(settings.retrieve_candidate_k, 20)
    settings.bm25_candidate_k = min(settings.bm25_candidate_k, 15)


apply_rag_profile()


def cors_origins_list() -> list[str]:
    raw = settings.cors_origins.strip()
    if raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]
