from __future__ import annotations

import inspect
import sys
import types
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, patch

import pytest

import flyte
from flyte._build import ImageBuild
from flyte._deploy import (
    DeploymentPlan,
    _build_image_bg,
    _build_images,
    _check_duplicate_env,
    _get_documentation_entity,
    _recursive_discover,
    _update_interface_inputs_and_outputs_docstring,
    build_images,
    plan_deploy,
)
from flyte._docstring import Docstring
from flyte._image import CodeBundleLayer, resolve_code_bundle_layer
from flyte._internal.imagebuild.image_builder import ImageCache
from flyte._internal.runtime.types_serde import transform_native_to_typed_interface
from flyte.models import NativeInterface


def test_get_description_entity_both_descriptions_truncated():
    # Create descriptions that exceed both limits
    env_desc = "a" * 300
    short_desc = "c" * 300
    long_desc = "d" * 3000
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10", description=env_desc)

    @env.task()
    async def task_both_exceed(x: int) -> int:
        return x * 2

    # Create a mock docstring with both descriptions exceeding limits
    mock_parsed_docstring = Mock()
    mock_parsed_docstring.short_description = short_desc
    mock_parsed_docstring.long_description = long_desc

    docstring = Docstring()
    docstring._parsed_docstring = mock_parsed_docstring
    # Use replace since NativeInterface is frozen
    task_both_exceed.interface = replace(task_both_exceed.interface, docstring=docstring)

    result = _get_documentation_entity(task_both_exceed)

    # Verify truncation with ...(tr.) suffix
    assert env.description == "a" * 247 + "...(tr.)"
    assert len(env.description) == 255
    assert result.short_description == "c" * 247 + "...(tr.)"
    assert len(result.short_description) == 255
    assert result.long_description == "d" * 2040 + "...(tr.)"
    assert len(result.long_description) == 2048


def test_get_description_entity_none_values():
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10")

    @env.task()
    async def task_no_docstring(x: int) -> int:
        return x * 2

    # Create a mock docstring with None descriptions
    mock_parsed_docstring = Mock()
    mock_parsed_docstring.short_description = None
    mock_parsed_docstring.long_description = None

    docstring = Docstring()
    docstring._parsed_docstring = mock_parsed_docstring
    # Use replace since NativeInterface is frozen
    task_no_docstring.interface = replace(task_no_docstring.interface, docstring=docstring)

    result = _get_documentation_entity(task_no_docstring)

    # Verify None values are handled correctly
    # Note: protobuf converts None to empty string for string fields
    assert env.description is None
    assert result.short_description == ""
    assert result.long_description == ""


def test_update_interface_with_docstring():
    docstring_text = """
    A test function.

    Args:
        x: The input value
        y: Another input

    Returns:
        The result
    """

    interface = NativeInterface(
        inputs={"x": (int, inspect.Parameter.empty), "y": (str, inspect.Parameter.empty)},
        outputs={"o0": int},
        docstring=Docstring(docstring=docstring_text),
    )

    typed_interface = transform_native_to_typed_interface(interface)

    # Before update, descriptions should be empty
    inputs_dict = {entry.key: entry.value for entry in typed_interface.inputs.variables}
    assert inputs_dict["x"].description == ""
    assert inputs_dict["y"].description == ""

    # Update descriptions
    result = _update_interface_inputs_and_outputs_docstring(typed_interface, interface)

    # After update, descriptions should be set
    result_inputs = {entry.key: entry.value for entry in result.inputs.variables}
    assert result_inputs["x"].description == "The input value"
    assert result_inputs["y"].description == "Another input"


def test_update_interface_no_docstring():
    interface = NativeInterface(
        inputs={"x": (int, inspect.Parameter.empty)},
        outputs={"o0": int},
        docstring=None,
    )

    typed_interface = transform_native_to_typed_interface(interface)
    result = _update_interface_inputs_and_outputs_docstring(typed_interface, interface)

    # Descriptions should remain empty
    result_inputs = {entry.key: entry.value for entry in result.inputs.variables}
    result_outputs = {entry.key: entry.value for entry in result.outputs.variables}
    assert result_inputs["x"].description == ""
    assert result_outputs["o0"].description == ""


def test_update_interface_empty_interface():
    interface = NativeInterface(
        inputs={},
        outputs={},
        docstring=None,
    )

    typed_interface = transform_native_to_typed_interface(interface)
    result = _update_interface_inputs_and_outputs_docstring(typed_interface, interface)

    # Should not raise any errors
    assert len(result.inputs.variables) == 0
    assert len(result.outputs.variables) == 0


def test_update_interface_partial_descriptions():
    docstring_text = """
    A test function.

    Args:
        x: The input value
    """

    interface = NativeInterface(
        inputs={"x": (int, inspect.Parameter.empty), "y": (str, inspect.Parameter.empty)},
        outputs={"o0": int},
        docstring=Docstring(docstring=docstring_text),
    )

    typed_interface = transform_native_to_typed_interface(interface)
    result = _update_interface_inputs_and_outputs_docstring(typed_interface, interface)

    # Only x should have description
    result_inputs = {entry.key: entry.value for entry in result.inputs.variables}
    assert result_inputs["x"].description == "The input value"
    assert result_inputs["y"].description == ""


def test_update_interface_mismatched_names():
    docstring_text = """
    A test function.

    Args:
        name: The user's name
        age: The user's age
    """

    # Interface has different parameter names
    interface = NativeInterface(
        inputs={"x": (int, inspect.Parameter.empty), "y": (str, inspect.Parameter.empty)},
        outputs={"o0": int},
        docstring=Docstring(docstring=docstring_text),
    )

    typed_interface = transform_native_to_typed_interface(interface)
    result = _update_interface_inputs_and_outputs_docstring(typed_interface, interface)

    # Descriptions should not be set (names don't match)
    result_inputs = {entry.key: entry.value for entry in result.inputs.variables}
    assert result_inputs["x"].description == ""
    assert result_inputs["y"].description == ""


# ---------------------------------------------------------------------------
# _check_duplicate_env / plan_deploy — duplicate detection tests
# ---------------------------------------------------------------------------


def _make_module(name: str, file: str, env: flyte.TaskEnvironment) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = file
    mod.env = env
    return mod


@pytest.fixture()
def dual_import_envs():
    """Two distinct env objects with the same name, each registered to a module
    that points at the same physical file (the classic src/ dual-import scenario)."""
    env1 = flyte.TaskEnvironment(name="my_env", image="python:3.10")
    env2 = flyte.TaskEnvironment(name="my_env", image="python:3.10")
    mod1 = _make_module("my_module.envs", "/project/src/my_module/envs.py", env1)
    mod2 = _make_module("src.my_module.envs", "/project/src/my_module/envs.py", env2)
    modules = {"my_module.envs": mod1, "src.my_module.envs": mod2}
    return env1, env2, modules


def test_check_duplicate_env_dual_import(dual_import_envs):
    """Same physical file imported under two module names → dual-import hint."""
    env1, env2, modules = dual_import_envs
    with (
        patch.dict(sys.modules, modules),
        patch("flyte._deploy.os.path.samefile", return_value=True),
        pytest.raises(ValueError, match="imported twice under different module names"),
    ):
        _check_duplicate_env(env1, env2)


def test_check_duplicate_env_true_duplicate():
    """Two envs with the same name from genuinely different files → plain duplicate error."""
    env1 = flyte.TaskEnvironment(name="my_env", image="python:3.10")
    env2 = flyte.TaskEnvironment(name="my_env", image="python:3.10")
    mod1 = _make_module("module_a.envs", "/project/module_a/envs.py", env1)
    mod2 = _make_module("module_b.envs", "/project/module_b/envs.py", env2)
    with (
        patch.dict(sys.modules, {"module_a.envs": mod1, "module_b.envs": mod2}),
        patch("flyte._deploy.os.path.samefile", return_value=False),
        pytest.raises(ValueError, match="Duplicate environment name 'my_env'"),
    ):
        _check_duplicate_env(env1, env2)


def test_plan_deploy_dual_import_raises(dual_import_envs):
    """plan_deploy surfaces the dual-import error when the same env name appears twice."""
    env1, env2, modules = dual_import_envs
    with (
        patch.dict(sys.modules, modules),
        patch("flyte._deploy.os.path.samefile", return_value=True),
        pytest.raises(ValueError, match="imported twice under different module names"),
    ):
        plan_deploy(env1, env2)


def test_plan_deploy_explicit_dep_no_false_positive():
    """flyte.deploy(env_a, env_b) where env_a.depends_on=[env_b] must not raise."""
    env_b = flyte.TaskEnvironment(name="b", image="python:3.10")
    env_a = flyte.TaskEnvironment(name="a", image="python:3.10", depends_on=[env_b])
    plans = plan_deploy(env_a, env_b)  # must not raise
    assert len(plans) == 1  # env_b already covered, no second plan


def test_recursive_discover_dual_import_raises(dual_import_envs):
    """_recursive_discover surfaces the dual-import error via the identity guard."""
    env1, env2, modules = dual_import_envs
    with (
        patch.dict(sys.modules, modules),
        patch("flyte._deploy.os.path.samefile", return_value=True),
        pytest.raises(ValueError, match="imported twice under different module names"),
    ):
        _recursive_discover({"my_env": env1}, env2)


def test_check_duplicate_env_dual_import_shows_distinct_sys_modules_keys():
    """When both module objects share the same ``__name__`` (e.g. both were created via
    ``importlib.util.spec_from_file_location`` with the file stem), the error message must still
    show the *sys.modules keys*, which are the distinguishing import paths the user can act on.

    Reproduces FLYTE-SDK-2S: the error message showed
    ``imported twice under different module names ('multi_status' and 'multi_status')`` —
    identical names — because the code was reading ``module.__name__`` instead of the
    sys.modules dict key.
    """
    env1 = flyte.TaskEnvironment(name="multi_status_demo", image="python:3.10")
    env2 = flyte.TaskEnvironment(name="multi_status_demo", image="python:3.10")

    # Both module objects intentionally have ``__name__ == "multi_status"`` (the file stem).
    # Their sys.modules keys are different — that's what the error should show.
    mod1 = _make_module("multi_status", "/project/examples/basics/multi_status.py", env1)
    mod2 = _make_module("multi_status", "/project/examples/basics/multi_status.py", env2)

    modules = {
        "multi_status": mod1,
        "examples.basics.multi_status": mod2,
    }

    with (
        patch.dict(sys.modules, modules),
        patch("flyte._deploy.os.path.samefile", return_value=True),
        pytest.raises(ValueError) as excinfo,
    ):
        _check_duplicate_env(env1, env2)

    msg = str(excinfo.value)
    assert "'multi_status'" in msg
    assert "'examples.basics.multi_status'" in msg
    # Guardrail against the regression: the two displayed import names must differ.
    assert "('multi_status' and 'multi_status')" not in msg


# ---------------------------------------------------------------------------
# _build_image_bg and _build_images tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_image_bg_captures_remote_run_url():
    """RunIdentifierData is extracted from remote_run when using the remote builder."""
    from flyte._internal.imagebuild.image_builder import RunIdentifierData

    image = flyte.Image.from_base("python:3.10")
    mock_run_id = Mock()
    mock_run_id.org = "my-org"
    mock_run_id.project = "my-project"
    mock_run_id.domain = "development"
    mock_run_id.name = "abc123"
    mock_run = Mock()
    mock_run.pb2.action.id.run = mock_run_id
    mock_result = ImageBuild(uri="registry/my-image:sha256abc", remote_run=mock_run)

    with patch("flyte._build.build") as mock_build:
        mock_build.aio = AsyncMock(return_value=mock_result)
        env_name, uri, run_id_data = await _build_image_bg("my-env", image)

    assert env_name == "my-env"
    assert uri == "registry/my-image:sha256abc"
    assert run_id_data == RunIdentifierData(org="my-org", project="my-project", domain="development", name="abc123")


@pytest.mark.asyncio
async def test_build_image_bg_no_url_for_local_build():
    """Build URL is None when using the local builder (remote_run is None)."""
    image = flyte.Image.from_base("python:3.10")
    mock_result = ImageBuild(uri="registry/my-image:sha256abc", remote_run=None)

    with patch("flyte._build.build") as mock_build:
        mock_build.aio = AsyncMock(return_value=mock_result)
        _env_name, _uri, build_url = await _build_image_bg("my-env", image)

    assert build_url is None


@pytest.mark.asyncio
async def test_build_images_stores_build_run_urls_in_cache():
    """build_run_ids in ImageCache is populated when remote builder provides a run identifier."""
    from flyte._internal.imagebuild.image_builder import RunIdentifierData

    flyte.init()
    image = flyte.Image.from_base("python:3.10")
    env = flyte.TaskEnvironment(name="my-env", image=image)
    plan = DeploymentPlan(envs={"my-env": env})

    mock_run_id = Mock()
    mock_run_id.org = "my-org"
    mock_run_id.project = "my-project"
    mock_run_id.domain = "development"
    mock_run_id.name = "abc123"
    mock_run = Mock()
    mock_run.pb2.action.id.run = mock_run_id
    mock_result = ImageBuild(uri="registry/my-image:sha256abc", remote_run=mock_run)

    with patch("flyte._build.build") as mock_build:
        mock_build.aio = AsyncMock(return_value=mock_result)
        cache: ImageCache = await _build_images(plan)

    assert cache.image_lookup["my-env"] == "registry/my-image:sha256abc"
    assert cache.build_run_ids["my-env"] == RunIdentifierData(
        org="my-org", project="my-project", domain="development", name="abc123"
    )


@pytest.mark.asyncio
async def test_build_images_no_build_run_urls_for_local_build():
    """build_run_ids in ImageCache is empty when local builder is used."""
    flyte.init()
    image = flyte.Image.from_base("python:3.10")
    env = flyte.TaskEnvironment(name="my-env", image=image)
    plan = DeploymentPlan(envs={"my-env": env})

    mock_result = ImageBuild(uri="registry/my-image:sha256abc", remote_run=None)

    with patch("flyte._build.build") as mock_build:
        mock_build.aio = AsyncMock(return_value=mock_result)
        cache: ImageCache = await _build_images(plan)

    assert cache.image_lookup["my-env"] == "registry/my-image:sha256abc"
    assert cache.build_run_ids == {}


# ---------------------------------------------------------------------------
# resolve_code_bundle_layer across depends_on environments
# ---------------------------------------------------------------------------


def test_resolve_covers_depends_on_envs():
    """plan_deploy + resolve_code_bundle_layer must strip CodeBundleLayer from
    depends_on environments, not just the parent env.

    Regression test: _run_remote and _run_hybrid previously only resolved
    parent_env.image, leaving depends_on images with root_dir=None, which
    caused 'root_dir not set for CodeBundleLayer' when computing image hashes.
    """
    from pathlib import Path

    dep_image = flyte.Image.from_base("python:3.10").clone(registry="r", name="dep", extendable=True).with_code_bundle()
    parent_image = (
        flyte.Image.from_base("python:3.10").clone(registry="r", name="parent", extendable=True).with_code_bundle()
    )

    env_dep = flyte.TaskEnvironment(name="dep", image=dep_image)
    env_parent = flyte.TaskEnvironment(name="parent", image=parent_image, depends_on=[env_dep])

    # Simulate exactly what the fixed _run_remote / _run_hybrid does.
    for _env in plan_deploy(env_parent)[0].envs.values():
        from flyte._image import Image

        if isinstance(_env.image, Image):
            _env.image = resolve_code_bundle_layer(_env.image, "loaded_modules", Path("/tmp"))

    # Both images must have their CodeBundleLayer stripped.
    assert not any(isinstance(layer, CodeBundleLayer) for layer in env_parent.image._layers), (
        "parent env still has CodeBundleLayer after resolution"
    )
    assert not any(isinstance(layer, CodeBundleLayer) for layer in env_dep.image._layers), (
        "depends_on env still has CodeBundleLayer after resolution — regression"
    )


# ---------------------------------------------------------------------------
# build_images resolves CodeBundleLayer (regression for flyte build CLI)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_images_resolves_code_bundle_layer_default_copy_style():
    """`flyte build` would previously raise 'root_dir not set for CodeBundleLayer'
    on any image that used `.with_code_bundle()`, because build_images did not
    run resolve_code_bundle_layer the way apply() does. With the default
    copy_style='loaded_modules', the bundle layer must be stripped before build.
    """
    flyte.init()

    image = flyte.Image.from_base("python:3.10").clone(registry="r", name="img", extendable=True).with_code_bundle()
    env = flyte.TaskEnvironment(name="e", image=image)

    with patch("flyte._build.build") as mock_build:
        mock_build.aio = AsyncMock(return_value=ImageBuild(uri="registry/img:abc", remote_run=None))
        await build_images.aio(env)

    assert not any(isinstance(layer, CodeBundleLayer) for layer in env.image._layers), (
        "CodeBundleLayer not stripped at copy_style='loaded_modules'"
    )


@pytest.mark.asyncio
async def test_build_images_resolves_code_bundle_layer_copy_style_none():
    """At copy_style='none', CodeBundleLayer is resolved in place (not stripped)
    so the source gets baked into the image via a COPY instruction.
    """
    flyte.init()

    image = flyte.Image.from_base("python:3.10").clone(registry="r", name="img", extendable=True).with_code_bundle()
    env = flyte.TaskEnvironment(name="e", image=image)

    with patch("flyte._build.build") as mock_build:
        mock_build.aio = AsyncMock(return_value=ImageBuild(uri="registry/img:abc", remote_run=None))
        await build_images.aio(env, copy_style="none")

    # Layer should still be present (resolved, with root_dir populated), not stripped.
    bundle_layers = [layer for layer in env.image._layers if isinstance(layer, CodeBundleLayer)]
    assert len(bundle_layers) == 1, "CodeBundleLayer should remain (resolved) at copy_style='none'"
    assert bundle_layers[0].root_dir is not None, "resolved CodeBundleLayer must have root_dir set"


@pytest.mark.asyncio
async def test_apply_unpicklable_env_raises_click_exception():
    """If the user's envs cannot be serialized, apply() should surface a friendly ClickException."""
    import pathlib

    import click

    from flyte._deploy import apply

    class _Unserializable:
        def __reduce__(self):
            raise TypeError("Cannot serialize objects that map to tty handles")

    plan = DeploymentPlan(envs={"e": _Unserializable()}, version=None)  # type: ignore[dict-item]

    fake_bundle = Mock()
    fake_bundle.computed_version = "test-bundle-version"

    fake_cfg = Mock()
    fake_cfg.root_dir = pathlib.Path("/tmp")
    fake_cfg.images = {}
    fake_cfg.project = "p"
    fake_cfg.domain = "d"
    fake_cfg.org = "o"

    with (
        patch("flyte._initialize.is_initialized", return_value=True),
        patch("flyte._deploy.get_init_config", return_value=fake_cfg),
        patch("flyte._deploy._build_images", new=AsyncMock(return_value={})),
        patch("flyte._code_bundle._includes.collect_env_include_files", return_value=[]),
        patch("flyte._code_bundle.build_code_bundle", new=AsyncMock(return_value=fake_bundle)),
    ):
        with pytest.raises(click.ClickException) as excinfo:
            await apply(plan, copy_style="loaded_modules", dryrun=True)
    assert "unpicklable" in excinfo.value.message
    assert "version=" in excinfo.value.message
