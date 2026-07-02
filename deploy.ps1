# 部署快捷脚本（含安全检查）
#   .\deploy.ps1           构建并启动
#   .\deploy.ps1 -Down     停止
#   .\deploy.ps1 -Logs     backend 日志

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
    Write-Host "[ERROR] 缺少 backend\.env" -ForegroundColor Red
    Write-Host "  copy backend\.env.example backend\.env 并填入 DEEPSEEK_API_KEY" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path ".env")) {
    Write-Host "[ERROR] 缺少项目根 .env（含 APP_API_KEY）" -ForegroundColor Red
    Write-Host "  copy .env.example .env" -ForegroundColor Yellow
    Write-Host "  python -c `"import secrets; print(secrets.token_urlsafe(32))`"" -ForegroundColor Yellow
    exit 1
}

$rootEnv = Get-Content ".env" -Raw
if ($rootEnv -notmatch "APP_API_KEY=\s*\S+" -or $rootEnv -match "APP_API_KEY=change-me") {
    Write-Host "[ERROR] 请在根 .env 设置随机 APP_API_KEY（见 docs/SECURITY.md）" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "backend\data\chroma")) {
    Write-Host "[WARN] 未找到 backend\data\chroma，请先：cd backend && python scripts/build_index.py" -ForegroundColor Yellow
}

if ($rootEnv -match "NGINX_BASIC_AUTH=\s*on" -and -not (Test-Path "deploy\htpasswd")) {
    Write-Host "[ERROR] NGINX_BASIC_AUTH=on 但缺少 deploy\htpasswd" -ForegroundColor Red
    Write-Host "  htpasswd -cb deploy\htpasswd admin 你的密码" -ForegroundColor Yellow
    exit 1
}

Write-Host "=== Docker Compose 安全部署 ===" -ForegroundColor Cyan
docker compose up -d --build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "访问: http://localhost:8080" -ForegroundColor Green
Write-Host "健康: http://localhost:8080/api/health" -ForegroundColor Green
Write-Host "安全说明: docs/SECURITY.md" -ForegroundColor Gray
