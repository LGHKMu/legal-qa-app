"""CI Recall 门禁：构建索引 + 跑 Agent 评测 + 阈值校验。

用法:
  cd backend
  python scripts/recall_gate.py                    # 默认 ci_no_llm
  python scripts/recall_gate.py --profile ci_full    # 需 DEEPSEEK_API_KEY
  python scripts/recall_gate.py --skip-build         # 索引已存在时跳过构建
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

BACKEND = Path(__file__).resolve().parent.parent
GATE_FILE = BACKEND / "data" / "recall_gate.yaml"
PYTHON = sys.executable


def load_gate_config() -> dict:
    with open(GATE_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_profile_env(profile: dict) -> None:
    for key, value in (profile.get("env") or {}).items():
        os.environ[key] = str(value)


def build_index() -> int:
    print("=== 构建向量库 / BM25 索引 ===", flush=True)
    return subprocess.call([PYTHON, "scripts/build_index.py"], cwd=BACKEND)


def run_compare_gate(profile: dict, cfg: dict) -> int:
    compare = BACKEND / "scripts" / "compare_rag.py"
    cmd = [PYTHON, str(compare), "--retrieval-only", "--eval-file", cfg["eval_file"]]
    top_k = profile.get("top_k") or cfg.get("top_k")
    if top_k:
        cmd.extend(["--top-k", str(top_k)])

    command = profile.get("command", "compare_agent")
    if command == "compare_agent":
        cmd.append("--compare-agent")
    elif command == "hybrid_no_rewrite":
        cmd.append("--no-rewrite")
    else:
        raise ValueError(f"未知 command: {command}")

    gate_mode = profile.get("gate_mode", "retrieval_agent")
    min_recall = profile["min_recall"]
    cmd.extend(["--gate-mode", gate_mode, "--min-recall", str(min_recall)])

    print("=== Recall 门禁评测 ===", flush=True)
    print(">", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=BACKEND)


def main() -> int:
    parser = argparse.ArgumentParser(description="CI Recall 门禁")
    parser.add_argument(
        "--profile",
        default=os.environ.get("RECALL_GATE_PROFILE", "ci_no_llm"),
        help="recall_gate.yaml 中的 profile 名（默认 ci_no_llm）",
    )
    parser.add_argument("--skip-build", action="store_true", help="跳过 build_index")
    args = parser.parse_args()

    cfg = load_gate_config()
    profiles = cfg.get("profiles") or {}
    profile = profiles.get(args.profile)
    if not profile:
        print(f"未知 profile: {args.profile}", file=sys.stderr)
        return 2

    if profile.get("requires_api_key") and not os.environ.get("DEEPSEEK_API_KEY"):
        print(
            f"profile {args.profile} 需要 DEEPSEEK_API_KEY，跳过门禁",
            flush=True,
        )
        return 0

    apply_profile_env(profile)
    print(f"Profile: {args.profile} — {profile.get('description', '')}", flush=True)
    print(
        f"阈值: {profile.get('gate_mode')} Recall@{cfg.get('top_k', 5)} "
        f">= {profile['min_recall']:.0%}",
        flush=True,
    )

    if not args.skip_build:
        code = build_index()
        if code:
            return code

    return run_compare_gate(profile, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
