"""Tests for LOG_LEVEL / USER_LOG_LEVEL forwarding in _Runner._build_env_dict.

Regression: a shell-set USER_LOG_LEVEL was dropped on remote runs because, unlike
LOG_LEVEL, it had no fallback to the logger's effective level when not explicitly
passed through with_runcontext.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

from flyte._run import _Runner

_NO_SYNC_CFG = SimpleNamespace(sync_local_sys_paths=False, root_dir=".")


def _env(**runner_kwargs):
    with patch("flyte._run.get_init_config", return_value=_NO_SYNC_CFG):
        return _Runner(**runner_kwargs)._build_env_dict()


def test_user_log_level_falls_back_to_logger_level():
    # Not passed explicitly -> should still be present (mirrors LOG_LEVEL behavior).
    with patch("flyte._run.user_logger.getEffectiveLevel", return_value=logging.DEBUG):
        env = _env()
    assert env["USER_LOG_LEVEL"] == str(logging.DEBUG)


def test_explicit_user_log_level_wins_over_logger_level():
    with patch("flyte._run.user_logger.getEffectiveLevel", return_value=logging.INFO):
        env = _env(user_log_level=logging.ERROR)
    assert env["USER_LOG_LEVEL"] == str(logging.ERROR)


def test_user_log_level_from_env_vars_is_preserved():
    env = _env(env_vars={"USER_LOG_LEVEL": "10"})
    assert env["USER_LOG_LEVEL"] == "10"


def test_both_log_levels_always_present():
    env = _env()
    assert "LOG_LEVEL" in env
    assert "USER_LOG_LEVEL" in env
