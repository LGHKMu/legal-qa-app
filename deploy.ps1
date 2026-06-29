# Docker 部署快捷脚本
# 用法：
#   .\deploy.ps1           构建并启动
#   .\deploy.ps1 -Down     停止
#   .\deploy.ps1 -Logs     查看 backend 日志

param(
    [switch]$Down,
    [switch]$Logs
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Down) {
    docker compose down
    exit $LASTEXITCODE
}

if ($Logs) {
    docker compose logs -f backend
    exit $LASTEXITCODE
}

if (-not (Test-Path "backend\.env")) {
    Write-Host "[ERROR] 缺少 backend\.env，请先：copy backend\.env.example backend\.env 并填入 DEEPSEEK_API_KEY" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "backend\data\chroma")) {
    Write-Host "[WARN] 未找到 backend\data\chroma，请先运行：cd backend && python scripts/build_index.py" -ForegroundColor Yellow
}

Write-Host "=== Docker Compose 部署 ===" -ForegroundColor Cyan
docker compose up -d --build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "访问: http://localhost:8080 （或 .env 中 APP_PORT）" -ForegroundColor Green
Write-Host "健康: http://localhost:8080/api/health" -ForegroundColor Green
Write-Host "日志: docker compose logs -f backend" -ForegroundColor Gray
