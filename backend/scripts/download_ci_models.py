"""CI 预下载 embedding 与 reranker 到 data/models/（无需 API Key）。"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND / "data" / "models"

# (repo_id, local_subdir)
CI_MODELS = (
    ("BAAI/bge-small-zh-v1.5", "bge-small-zh-v1.5"),
    ("BAAI/bge-reranker-base", "bge-reranker-base"),
)


def _has_weights(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def download_one(repo_id: str, target: Path) -> None:
    if _has_weights(target):
        print(f"已存在，跳过: {target}", flush=True)
        return
    from huggingface_hub import snapshot_download

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载 {repo_id} -> {target}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target))
    if not _has_weights(target):
        raise RuntimeError(f"下载完成但缺少权重: {target}")


def main() -> int:
    errors: list[str] = []
    for repo_id, subdir in CI_MODELS:
        try:
            download_one(repo_id, MODELS_DIR / subdir)
        except Exception as exc:
            errors.append(f"{repo_id}: {exc}")
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print("CI 模型就绪", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
