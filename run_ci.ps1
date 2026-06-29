# 本地 CI 一键脚本（Windows PowerShell）
# 用法：在项目根目录 legal-qa-app/ 下执行
#   .\run_ci.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== legal-qa-app 本地 CI ===" -ForegroundColor Cyan
Set-Location backend
python scripts/run_all_tests.py --ci
exit $LASTEXITCODE
