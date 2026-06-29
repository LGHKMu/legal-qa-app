# Cascade 混合检索评测（Rerank 默认已开启，此脚本仅显式设置本地模型路径）
# 用法: cd backend; .\scripts\run_with_rerank.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$env:RERANK_ENABLED = "true"
$env:RERANK_MODEL_PATH = "./data/models/bge-reranker-base"
$env:RERANK_LOCAL_ONLY = "true"
$env:BM25_ENABLED = "true"

Write-Host "Cascade 混合检索评测..." -ForegroundColor Cyan
python scripts/compare_rag.py --compare-rewrite --retrieval-only
