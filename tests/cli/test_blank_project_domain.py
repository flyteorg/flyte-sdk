"""Regression tests for FLYTE-SDK-3A: a blank --project/--domain must not override config.

`flyte run --project "$PROJECT" --domain "$DOMAIN"` with unset shell variables hands the
CLI empty strings. Those used to be treated as explicit values, overriding the config file
and travelling all the way to CreateUploadLocation, where the backend answered
"failed to validate project: invalid_argument: id is required". That surfaced as a
RuntimeSystemError and was reported to Sentry as an SDK crash rather than as the user's
missing configuration.

The normalization lives on the click option, not in `CLIConfig.init`, because only a value
typed on the command line is ambiguous. `flyte get/create/delete secret` call `init` with a
literal `""` to select the org-level scope, and that has to survive -- the second class below
pins it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
import pytest

from flyte.cli._common import CLIConfig, CommandBase, blank_option_to_none
from flyte.config._config import Config, TaskConfig


def _make_cli_config(config: Config | None = None) -> CLIConfig:
    ctx = MagicMock(spec=click.Context)
    return CLIConfig(config=config or Config(), ctx=ctx)


CONFIGURED = Config(task=TaskConfig(project="from-config", domain="from-config-domain"))


def _invoke(args: list[str], config: Config) -> Config:
    """Run a `CommandBase` command with `args` and return the Config handed to init_from_config.

    Goes through click so the option callback runs, which is where the blank is normalized.
    """
    captured = {}

    @click.command(cls=CommandBase)
    @click.pass_context
    def cmd(ctx, project=None, domain=None):
        CLIConfig(config=config, ctx=ctx).init(project=project, domain=domain)

    with patch("flyte.cli._common.flyte") as mock_flyte:
        cmd.main(args=args, standalone_mode=False)
        captured["cfg"] = mock_flyte.init_from_config.call_args[0][0]
    return captured["cfg"]


class TestBlankProjectDomainOnTheCommandLine:
    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_cli_values_fall_back_to_config(self, blank):
        cfg = _invoke(["--project", blank, "--domain", blank], CONFIGURED)
        assert cfg.task.project == "from-config"
        assert cfg.task.domain == "from-config-domain"

    def test_explicit_values_still_override_config(self):
        cfg = _invoke(["--project", "explicit", "--domain", "explicit-domain"], CONFIGURED)
        assert cfg.task.project == "explicit"
        assert cfg.task.domain == "explicit-domain"

    def test_padded_values_are_stripped(self):
        cfg = _invoke(["--project", "  explicit  ", "--domain", "  explicit-domain  "], CONFIGURED)
        assert cfg.task.project == "explicit"
        assert cfg.task.domain == "explicit-domain"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_on_both_sides_leaves_project_unset(self, blank):
        """With nothing configured either, project stays None so the guard can raise.

        None is what `require_project_and_domain` reports as "Project must be provided",
        a user-kind InitializationError that is filtered out of Sentry.
        """
        cfg = _invoke(["--project", blank, "--domain", blank], Config())
        assert cfg.task.project is None
        assert cfg.task.domain is None

    def test_omitting_the_flags_entirely_falls_back_to_config(self):
        cfg = _invoke([], CONFIGURED)
        assert cfg.task.project == "from-config"
        assert cfg.task.domain == "from-config-domain"

    @pytest.mark.parametrize(
        "value, expected",
        [(None, None), ("", None), ("   ", None), ("\t", None), ("proj", "proj"), ("  proj  ", "proj")],
    )
    def test_callback_normalization(self, value, expected):
        assert blank_option_to_none(MagicMock(spec=click.Context), MagicMock(spec=click.Parameter), value) == expected


class TestOrgLevelSecretScopeIsPreserved:
    """`flyte get/create/delete secret` pass project="" to mean "org level", not "unset".

    Normalizing that blank away inside `CLIConfig.init` would make those three commands fall
    back to the config file's project/domain: the listing would silently change scope, and
    `--cluster-pool` would start failing outright, since `Secret._resolve_scope` rejects a
    request that carries both a cluster pool and a project/domain.
    """

    @patch("flyte.cli._common.flyte")
    def test_programmatic_blank_stays_blank(self, mock_flyte):
        cli = _make_cli_config(config=CONFIGURED)
        cli.init(project="", domain="")

        call_cfg = mock_flyte.init_from_config.call_args[0][0]
        assert call_cfg.task.project == ""
        assert call_cfg.task.domain == ""

    @patch("flyte.cli._common.flyte")
    def test_cluster_pool_scope_check_still_sees_an_empty_scope(self, mock_flyte):
        """The empty scope is what keeps the `--cluster-pool` guard from rejecting the request."""
        cli = _make_cli_config(config=CONFIGURED)
        cli.init(project="", domain="")

        call_cfg = mock_flyte.init_from_config.call_args[0][0]
        assert not call_cfg.task.project
        assert not call_cfg.task.domain

    @patch("flyte.cli._common.flyte")
    def test_programmatic_real_values_still_pass_through(self, mock_flyte):
        cli = _make_cli_config(config=CONFIGURED)
        cli.init(project="explicit", domain="explicit-domain")

        call_cfg = mock_flyte.init_from_config.call_args[0][0]
        assert call_cfg.task.project == "explicit"
        assert call_cfg.task.domain == "explicit-domain"
