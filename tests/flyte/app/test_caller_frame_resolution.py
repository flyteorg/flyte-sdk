"""Tests for ``AppEnvironment._caller_frame`` discovery.

The Flyte ``AppEnvResolver`` uses ``app_env._caller_frame.filename`` to figure
out which module to import in the deployed container so it can rebind the
``app_env`` variable for ``fserve``. If that filename ends up pointing at
SDK source (or worse, at the dataclass-synthesised ``__init__`` whose
filename is ``"<string>"``), the resolver-args are silently dropped from the
container command and ``fserve`` crashes with
``ValueError: No command provided to execute`` at boot.

This module locks in:

1. The parent ``AppEnvironment``'s walker always lands on a user-code frame
   (never inside the SDK, never on a synthesised ``"<string>"`` frame).
2. The same property holds when an env is built through a factory helper
   (single level *or* nested).
3. Every subclass that previously shipped a buggy 2-level override
   (MCP / FastAPI / Webhook / AgentChat) now inherits the correct walker.
4. The resolver's ``loader_args`` round-trips to ``"<module>:<varname>"`` so
   ``fserve`` can re-import the binding.
"""

from __future__ import annotations

import pathlib

import pytest

from flyte._internal.resolvers.app_env import AppEnvResolver
from flyte.app import AppEnvironment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


THIS_FILE = pathlib.Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent


# A module-level binding so the round-trip resolver test below can locate an
# AppEnvironment when it imports this test module. Without this binding, the
# resolver (which only sees module globals) would have nothing to match
# against and would raise.
MODULE_LEVEL_ENV = AppEnvironment(name="caller-frame-module-level-env")


def _make_inline_env() -> AppEnvironment:
    """Module-equivalent inline factory; same file, single frame above ctor."""
    return AppEnvironment(name="caller-frame-factory")


def _make_outer_env() -> AppEnvironment:
    """Nested factory: deliberately wraps another factory call."""
    return _make_inline_env()


def _make_subclass_factory(env_cls, **kwargs) -> AppEnvironment:
    """Subclass factory used by parametrised subclass tests."""
    return env_cls(**kwargs)


# ---------------------------------------------------------------------------
# Parent AppEnvironment
# ---------------------------------------------------------------------------


def test_caller_frame_module_level_landing():
    """A direct call at module scope lands on the test file's <module>-equivalent frame."""
    env = AppEnvironment(name="caller-frame-direct")
    assert env._caller_frame is not None
    assert env._caller_frame.filename == str(THIS_FILE)
    # Inside a pytest test function the immediate caller is the test, not <module>.
    assert env._caller_frame.function == "test_caller_frame_module_level_landing"


def test_caller_frame_factory_lands_in_user_module():
    """A factory helper still resolves to the user file (not into the SDK)."""
    env = _make_inline_env()
    assert env._caller_frame is not None
    assert env._caller_frame.filename == str(THIS_FILE)
    assert env._caller_frame.function == "_make_inline_env"


def test_caller_frame_nested_factory_lands_in_user_module():
    """Nested factories still resolve to the innermost user frame, never to '<string>'."""
    env = _make_outer_env()
    assert env._caller_frame is not None
    assert env._caller_frame.filename == str(THIS_FILE)
    # The inner call wins because that frame is closer to AppEnvironment.__post_init__.
    assert env._caller_frame.function == "_make_inline_env"


def test_caller_frame_filename_never_synthesised():
    """Regression guard: filename must never be ``<string>`` / ``<frozen>`` / empty."""
    env = AppEnvironment(name="caller-frame-synthesised-check")
    assert env._caller_frame is not None
    assert not env._caller_frame.filename.startswith("<")


def test_caller_frame_filename_outside_sdk_tree():
    """The walker must skip every SDK-internal frame so the resolver works."""
    env = AppEnvironment(name="caller-frame-sdk-skip")
    assert env._caller_frame is not None
    # The walker reports an absolute path; just check it isn't inside flyte's source.
    assert "/site-packages/flyte/" not in env._caller_frame.filename
    assert "/src/flyte/" not in env._caller_frame.filename


def test_resolver_loader_args_round_trip():
    """End-to-end: resolver produces ``<module>:<varname>`` for module-level binding."""
    loader_args = AppEnvResolver().loader_args(MODULE_LEVEL_ENV, THIS_DIR)
    module_name, var_name = loader_args.split(":")
    assert module_name == THIS_FILE.stem
    assert var_name == "MODULE_LEVEL_ENV"


# ---------------------------------------------------------------------------
# Subclasses that previously clobbered _caller_frame
# ---------------------------------------------------------------------------


def _import_or_skip(qualified_name: str):
    """Import a class lazily; skip the test if the optional extra isn't installed."""
    import importlib

    module_path, _, class_name = qualified_name.rpartition(".")
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        pytest.skip(f"optional dependency missing for {qualified_name}: {e}")
    return getattr(mod, class_name)


def _subclass_kwargs(env_cls):
    """Minimal constructor kwargs per subclass."""
    name_only = {"name": "caller-frame-subclass"}
    if env_cls.__name__ == "FastAPIAppEnvironment":
        import fastapi

        return {**name_only, "app": fastapi.FastAPI()}
    if env_cls.__name__ == "AgentChatAppEnvironment":

        class _StubAgent:  # implements the Agent protocol minimally
            tool_descriptions = ()

            async def run(self, *_a, **_kw):  # pragma: no cover - never invoked in this test
                yield ""

        return {**name_only, "agent": _StubAgent()}
    return name_only


@pytest.mark.parametrize(
    "qualified_name",
    [
        "flyte.ai.mcp.FlyteMCPAppEnvironment",
        "flyte.app.extras.FastAPIAppEnvironment",
        "flyte.app.extras._webhook_app.FlyteWebhookAppEnvironment",
        "flyte.ai.chat.AgentChatAppEnvironment",
    ],
)
def test_subclasses_caller_frame_lands_in_user_code(qualified_name):
    """All subclasses that previously had a buggy override now inherit the fix."""
    env_cls = _import_or_skip(qualified_name)
    env = env_cls(**_subclass_kwargs(env_cls))
    assert env._caller_frame is not None
    assert env._caller_frame.filename == str(THIS_FILE), (
        f"{qualified_name} reported filename {env._caller_frame.filename!r}; expected the test file (i.e. user code)."
    )
    assert not env._caller_frame.filename.startswith("<")


@pytest.mark.parametrize(
    "qualified_name",
    [
        "flyte.ai.mcp.FlyteMCPAppEnvironment",
        "flyte.app.extras.FastAPIAppEnvironment",
        "flyte.app.extras._webhook_app.FlyteWebhookAppEnvironment",
        "flyte.ai.chat.AgentChatAppEnvironment",
    ],
)
def test_subclasses_caller_frame_factory_pattern(qualified_name):
    """Subclasses keep the user-code landing even when built via a helper."""
    env_cls = _import_or_skip(qualified_name)
    env = _make_subclass_factory(env_cls, **_subclass_kwargs(env_cls))
    assert env._caller_frame is not None
    assert env._caller_frame.filename == str(THIS_FILE)
    assert env._caller_frame.function == "_make_subclass_factory"


# ---------------------------------------------------------------------------
# container_command must include resolver-args now that _caller_frame is correct
# ---------------------------------------------------------------------------


def test_container_cmd_includes_resolver_args_for_module_level_env():
    """``container_cmd`` must add ``--resolver`` / ``--resolver-args``.

    Without those flags, ``fserve`` can't import ``app_env`` and crashes at
    boot. This is the precise failure mode the underlying bug produced.
    """
    from flyte._internal.imagebuild.image_builder import ImageCache
    from flyte.models import CodeBundle, SerializationContext

    ctx = SerializationContext(
        org="org",
        project="proj",
        domain="dev",
        version="v0",
        root_dir=THIS_DIR,
        code_bundle=CodeBundle(tgz="bundle.tgz", computed_version="v0", destination="/code"),
        image_cache=ImageCache(image_lookup={}),
    )
    cmd = MODULE_LEVEL_ENV.container_cmd(ctx)
    assert "--resolver" in cmd, f"expected --resolver in {cmd}"
    assert "--resolver-args" in cmd, f"expected --resolver-args in {cmd}"
    args_index = cmd.index("--resolver-args")
    loader_args = cmd[args_index + 1]
    module_name, var_name = loader_args.split(":")
    assert module_name == THIS_FILE.stem
    assert var_name == "MODULE_LEVEL_ENV"
