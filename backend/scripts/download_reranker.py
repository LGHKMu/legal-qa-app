"""下载 BGE Reranker 到本地（国内推荐 ModelScope，无需访问 huggingface.co）。

用法:
  cd backend
  pip install modelscope
  python scripts/download_reranker.py

下载完成后在 .env 中设置:
  RERANK_ENABLED=true
  RERANK_MODEL_PATH=./data/models/bge-reranker-base
  RERANK_LOCAL_ONLY=true
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR

MODEL_ID = "BAAI/bge-reranker-base"
LOCAL_DIR = DATA_DIR / "models" / "bge-reranker-base"


def download_via_modelscope(target: Path) -> Path:
    from modelscope import snapshot_download

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在从 ModelScope 下载 {MODEL_ID} ...", flush=True)
    print(f"目标目录: {target}", flush=True)
    path = snapshot_download(MODEL_ID, local_dir=str(target))
    print(f"下载完成: {path}", flush=True)
    return Path(path)


def download_via_hf_mirror(target: Path) -> Path:
    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在从 HF 镜像下载 {MODEL_ID} ...", flush=True)
    path = snapshot_download(repo_id=MODEL_ID, local_dir=str(target))
    print(f"下载完成: {path}", flush=True)
    return Path(path)


def verify_model(path: Path) -> bool:
    required = ("config.json",)
    optional_weight = (
        "pytorch_model.bin",
        "model.safetensors",
    )
    if not path.exists():
        return False
    if not all((path / f).exists() for f in required):
        return False
    return any((path / f).exists() for f in optional_weight)


def main() -> None:
    if verify_model(LOCAL_DIR):
        print(f"模型已存在，跳过下载: {LOCAL_DIR}")
        return

    errors: list[str] = []
    for name, fn in (
        ("ModelScope", download_via_modelscope),
        ("HF 镜像 (hf-mirror.com)", download_via_hf_mirror),
    ):
        try:
            fn(LOCAL_DIR)
            if verify_model(LOCAL_DIR):
                print("\n请在 backend/.env 添加:")
                print("RERANK_ENABLED=true")
                print("RERANK_MODEL_PATH=./data/models/bge-reranker-base")
                print("RERANK_LOCAL_ONLY=true")
                return
            errors.append(f"{name}: 下载完成但缺少 config.json / 权重文件")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    print("所有下载方式均失败:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    print(
        "\n可手动打开 https://www.modelscope.cn/models/BAAI/bge-reranker-base 下载后解压到:",
        file=sys.stderr,
    )
    print(f"  {LOCAL_DIR}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
