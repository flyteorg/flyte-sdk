"""Tests for `flyte deploy` reached without a CLIConfig on the click context (FLYTE-SDK-8N)."""

import pathlib
from unittest.mock import Mock, patch

import click

from flyte.cli._common import CLIConfig, initialize_config
from flyte.cli._deploy import DeployArguments, DeployEnvCommand, DeployEnvRecursiveCommand


def _deploy_args() -> DeployArguments:
    return DeployArguments(project="p", domain="d")


def _deployment() -> Mock:
    return Mock(env_repr=list, table_repr=list)


def test_initialize_config_puts_the_config_it_creates_on_the_context():
    """The config built for a bare context is stored on it, not only returned."""
    ctx = click.Context(click.Command("x"))
    assert ctx.obj is None

    obj = initialize_config(ctx, project="p", domain="d")

    assert isinstance(obj, CLIConfig)
    assert ctx.obj is obj


def test_initialize_config_keeps_an_existing_context_config():
    """A context that already carries a config keeps that exact object."""
    ctx = click.Context(click.Command("x"))
    existing = Mock(spec=CLIConfig)
    ctx.obj = existing

    obj = initialize_config(ctx, project="p", domain="d")

    assert obj is existing
    assert ctx.obj is existing
    existing.init.assert_called_once()


def test_recursive_deploy_without_a_context_config():
    """
    `DeployEnvRecursiveCommand.invoke` used to read `ctx.obj` before `initialize_config` ran and
    then use it, so a context reached without the top-level group callback died at the first
    `obj.output_format` as `AttributeError: 'NoneType' object has no attribute 'output_format'`.
    """
    cmd = DeployEnvRecursiveCommand(path=pathlib.Path("."), deploy_args=_deploy_args(), name="x")
    ctx = click.Context(cmd)
    env = Mock()
    env.name = "e1"

    with (
        patch("flyte._environment.list_loaded_environments", return_value=[env]),
        patch("flyte.deploy", return_value=[_deployment()]),
    ):
        cmd.invoke(ctx)

    assert isinstance(ctx.obj, CLIConfig)


def test_env_deploy_without_a_context_config():
    """
    The same context state reached `DeployEnvCommand.invoke`, which called `ctx.obj.init(...)`
    directly and died as `AttributeError: 'NoneType' object has no attribute 'init'`.
    """
    env = Mock()
    env.name = "e1"
    cmd = DeployEnvCommand(env_name="e1", env=env, deploy_args=_deploy_args(), name="y")
    ctx = click.Context(cmd)

    with patch("flyte.deploy", return_value=[_deployment()]):
        cmd.invoke(ctx)

    assert isinstance(ctx.obj, CLIConfig)
