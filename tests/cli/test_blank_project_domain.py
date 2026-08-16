"""Regression tests for FLYTE-SDK-3A: a blank --project/--domain must not override config.

`flyte run --project "$PROJECT" --domain "$DOMAIN"` with unset shell variables hands the
CLI empty strings. Those used to be treated as explicit values, overriding the config file
and travelling all the way to CreateUploadLocation, where the backend answered
"failed to validate project: invalid_argument: id is required". That surfaced as a
RuntimeSystemError and was reported to Sentry as an SDK crash rather than as the user's
missing configuration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
import pytest

from flyte.cli._common import CLIConfig
from flyte.config._config import Config, TaskConfig


def _make_cli_config(config: Config | None = None) -> CLIConfig:
    ctx = MagicMock(spec=click.Context)
    return CLIConfig(config=config or Config(), ctx=ctx)


CONFIGURED = Config(task=TaskConfig(project="from-config", domain="from-config-domain"))


class TestBlankProjectDomain:
    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    @patch("flyte.cli._common.flyte")
    def test_blank_cli_values_fall_back_to_config(self, mock_flyte, blank):
        cli = _make_cli_config(config=CONFIGURED)
        cli.init(project=blank, domain=blank)

        call_cfg = mock_flyte.init_from_config.call_args[0][0]
        assert call_cfg.task.project == "from-config"
        assert call_cfg.task.domain == "from-config-domain"

    @patch("flyte.cli._common.flyte")
    def test_explicit_values_still_override_config(self, mock_flyte):
        cli = _make_cli_config(config=CONFIGURED)
        cli.init(project="explicit", domain="explicit-domain")

        call_cfg = mock_flyte.init_from_config.call_args[0][0]
        assert call_cfg.task.project == "explicit"
        assert call_cfg.task.domain == "explicit-domain"

    @patch("flyte.cli._common.flyte")
    def test_padded_values_are_stripped(self, mock_flyte):
        cli = _make_cli_config(config=CONFIGURED)
        cli.init(project="  explicit  ", domain="  explicit-domain  ")

        call_cfg = mock_flyte.init_from_config.call_args[0][0]
        assert call_cfg.task.project == "explicit"
        assert call_cfg.task.domain == "explicit-domain"

    @pytest.mark.parametrize("blank", ["", "   "])
    @patch("flyte.cli._common.flyte")
    def test_blank_on_both_sides_leaves_project_unset(self, mock_flyte, blank):
        """With nothing configured either, project stays None so the guard can raise.

        None is what `require_project_and_domain` reports as "Project must be provided",
        a user-kind InitializationError that is filtered out of Sentry.
        """
        cli = _make_cli_config()
        cli.init(project=blank, domain=blank)

        call_cfg = mock_flyte.init_from_config.call_args[0][0]
        assert call_cfg.task.project is None
        assert call_cfg.task.domain is None
