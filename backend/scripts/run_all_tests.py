"""一键运行测试。

用法:
  cd backend

  # CI 门禁（单元测试 + 前端 build，约 1–3 分钟，不需 API Key）
  python scripts/run_all_tests.py --ci

  # 快速全套（冒烟 + 诊断 + 口语题，约 8–12 分钟，需 API + 索引）
  python scripts/run_all_tests.py --quick

  # 完整全套（约 40–60 分钟）
  python scripts/run_all_tests.py --full

  # 仅冒烟（5 秒，不需 API）
  python scripts/run_all_tests.py --smoke-only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
FRONTEND = ROOT / "frontend"
PYTHON = sys.executable


def run_step(title: str, cmd: list[str], *, cwd: Path | None = None) -> int:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(">", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd or BACKEND)
    return result.returncode


def run_ci() -> int:
    failed = 0

    if shutil.which("pytest") is None:
        print("未找到 pytest，正在安装 requirements-dev.txt …")
        code = run_step(
            "[setup] pip install -r requirements-dev.txt",
            [PYTHON, "-m", "pip", "install", "-r", "requirements-dev.txt"],
        )
        if code:
            return code

    failed += run_step(
        "[1/2] 单元测试 pytest tests/unit",
        [PYTHON, "-m", "pytest", "tests/unit", "-q", "-m", "not live and not integration"],
    )

    npm = shutil.which("npm")
    if not npm:
        print("\n[WARN] 未找到 npm，跳过前端 build")
    elif not FRONTEND.is_dir():
        print("\n[WARN] 未找到 frontend/，跳过前端 build")
    else:
        failed += run_step(
            "[2/2] 前端 build",
            [npm, "run", "build"],
            cwd=FRONTEND,
        )

    print("\n" + "=" * 60)
    if failed:
        print(f"CI 未通过：{failed} 个步骤失败")
        return 1
    print("CI 全部通过")
    print("=" * 60)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="运行全部测试")
    parser.add_argument("--ci", action="store_true", help="CI 门禁：单元测试 + 前端 build")
    parser.add_argument("--quick", action="store_true", help="快速测试集")
    parser.add_argument("--full", action="store_true", help="完整 60 题测试")
    parser.add_argument("--smoke-only", action="store_true", help="仅冒烟")
    args = parser.parse_args()

    if args.ci:
        sys.exit(run_ci())

    if not any([args.quick, args.full, args.smoke_only]):
        args.quick = True

    scripts = BACKEND / "scripts"
    failed = 0

    if args.smoke_only:
        code = run_step(
            "[1/1] 冒烟测试（pytest 单元测试）",
            [PYTHON, "-m", "pytest", "tests/unit", "-q", "-m", "not live and not integration"],
        )
        sys.exit(code)

    failed += run_step(
        "[1/N] 单元测试",
        [PYTHON, "-m", "pytest", "tests/unit", "-q", "-m", "not live and not integration"],
    )

    failed += run_step(
        "[2/N] 关键题诊断 v05,v26,v41（baseline / 改写 / Cascade 混合）",
        [
            PYTHON,
            str(scripts / "test_rag_pipeline.py"),
            "--ids",
            "v05,v26,v41",
            "--output",
            "data/test_key_questions.json",
        ],
    )

    if args.quick:
        failed += run_step(
            "[3/N] 口语题集 q41-q60",
            [
                PYTHON,
                str(scripts / "test_rag_pipeline.py"),
                "--oral",
                "--output",
                "data/test_oral.json",
            ],
        )
        failed += run_step(
            "[4/N] 混合诊断 v25,v68",
            [PYTHON, str(scripts / "diagnose_hybrid_gap.py"), "--ids", "v25,v68"],
        )
    else:
        failed += run_step(
            "[3/N] 60 题 baseline | 改写 | 双路 RRF",
            [
                PYTHON,
                str(scripts / "compare_rag.py"),
                "--compare-rewrite",
                "--retrieval-only",
                "--output",
                "data/eval_rewrite_rrf.json",
            ],
        )
        failed += run_step(
            "[4/N] Cascade 混合诊断",
            [PYTHON, str(scripts / "diagnose_hybrid_gap.py"), "--ids", "v25,v68"],
        )

    print("\n" + "=" * 60)
    if failed:
        print(f"完成，{failed} 个步骤失败")
        sys.exit(1)
    print("全部测试通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
