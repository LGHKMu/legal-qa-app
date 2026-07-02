"""Recall 门禁逻辑单元测试。"""

from __future__ import annotations

import pytest

from scripts.compare_rag import (
    Summary,
    enforce_recall_gate,
    normalize_gate_mode,
    recall_for_gate_mode,
)


def test_normalize_gate_mode_aliases() -> None:
    assert normalize_gate_mode("agent") == "retrieval_agent"
    assert normalize_gate_mode("retrieval") == "rag_retrieval"


def test_recall_for_gate_mode() -> None:
    summaries = [
        Summary(mode="retrieval_agent", count=10, recall_at_k=0.75),
        Summary(mode="rag_retrieval", count=10, recall_at_k=0.706),
    ]
    assert recall_for_gate_mode(summaries, "agent") == 0.75
    assert recall_for_gate_mode(summaries, "rag_retrieval") == 0.706
    assert recall_for_gate_mode(summaries, "missing") is None


def test_enforce_recall_gate_pass() -> None:
    summaries = [Summary(mode="retrieval_agent", count=68, recall_at_k=0.75)]
    enforce_recall_gate(summaries, gate_mode="agent", min_recall=0.72)


def test_enforce_recall_gate_fail() -> None:
    summaries = [Summary(mode="retrieval_agent", count=68, recall_at_k=0.70)]
    with pytest.raises(SystemExit) as exc:
        enforce_recall_gate(summaries, gate_mode="retrieval_agent", min_recall=0.72)
    assert exc.value.code == 1
