"""Agent 改写开关单元测试。"""

from __future__ import annotations

from unittest.mock import patch

from agent.tools import _agent_rewrite


@patch("config.settings")
def test_agent_rewrite_respects_global_disable(mock_settings) -> None:
    mock_settings.query_rewrite_enabled = False
    assert _agent_rewrite(True) is False
    assert _agent_rewrite(None) is False


@patch("config.settings")
def test_agent_rewrite_explicit_false(mock_settings) -> None:
    mock_settings.query_rewrite_enabled = True
    assert _agent_rewrite(False) is False


@patch("config.settings")
def test_agent_rewrite_when_enabled(mock_settings) -> None:
    mock_settings.query_rewrite_enabled = True
    assert _agent_rewrite(True) is True
    assert _agent_rewrite(None) is True
