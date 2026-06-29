"""Embedding / Reranker 推理设备解析。"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_inference_device(setting: str) -> str:
    """auto → 有 CUDA 用 cuda，否则 cpu；也可显式 cuda / cuda:0 / cpu。"""
    value = (setting or "auto").strip().lower()
    if value == "auto":
        return _detect_cuda_or_cpu()
    if value.startswith("cuda"):
        if _cuda_available():
            return value
        logger.warning("INFERENCE_DEVICE=%s 但 CUDA 不可用，回退 CPU", setting)
        return "cpu"
    return value


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _detect_cuda_or_cpu() -> str:
    if _cuda_available():
        try:
            import torch

            logger.info("推理设备: cuda (%s)", torch.cuda.get_device_name(0))
        except Exception:
            logger.info("推理设备: cuda")
        return "cuda"
    logger.info("推理设备: cpu（未检测到 CUDA）")
    return "cpu"
