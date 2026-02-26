"""Tests for flyte.sandbox.create() and the _Sandbox class."""

import datetime
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pytest

import flyte
import flyte.sandbox
from flyte.extras._container import ContainerTask
from flyte.io import File
from flyte.models import NativeInterface
from flyte.sandbox._code_sandbox import ImageConfig, _Sandbox, create

# ---------------------------------------------------------------------------
# create() — argument validation
# ---------------------------------------------------------------------------


class TestCreateValidation:
    def test_code_and_command_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            create(code="x = 1", command=["echo", "hi"])

    def test_unsupported_input_type_rejected(self):
        with pytest.raises(TypeError, match="Unsupported input type"):
            create(name="sb", code="pass", inputs={"x": list})

    def test_unsupported_output_type_rejected(self):
        with pytest.raises(TypeError, match="Unsupported output type"):
            create(name="sb", code="pass", outputs={"result": dict})

    def test_code_only_is_valid(self):
        sb = create(name="sb", code="x = 1")
        assert sb.code == "x = 1"
        assert sb.command is None

    def test_command_only_is_valid(self):
        sb = create(name="sb", command=["echo", "hi"])
        assert sb.command == ["echo", "hi"]
        assert sb.code is None

    def test_neither_code_nor_command_is_valid(self):
        # Neither is required — e.g. image already has an ENTRYPOINT
        sb = create(name="sb")
        assert sb.code is None
        assert sb.command is None

    def test_defaults(self):
        sb = create(name="my-sandbox")
        assert sb.auto_io is True
        assert sb.retries == 0
        assert sb.timeout is None
        assert sb.env_vars is None
        assert sb.secrets is None
        assert sb.cache == "auto"
        assert sb.packages == []
        assert sb.system_packages == []
        assert sb.additional_commands == []

    def test_inputs_defaults_to_empty_dict(self):
        sb = create(name="sb", code="pass")
        assert sb.inputs == {}

    def test_outputs_defaults_to_empty_dict(self):
        sb = create(name="sb", code="pass")
        assert sb.outputs == {}

    def test_explicit_inputs_and_outputs(self):
        sb = create(
            name="sb",
            code="pass",
            inputs={"x": int, "y": str},
            outputs={"result": float},
        )
        assert sb.inputs == {"x": int, "y": str}
        assert sb.outputs == {"result": float}

    def test_image_skips_build(self):
        sb = create(name="sb", code="pass", image="myregistry/myimage:latest")
        assert sb.image == "myregistry/myimage:latest"

    def test_image_config(self):
        cfg = ImageConfig(registry="gcr.io/my-project", python_version=(3, 11))
        sb = create(name="sb", code="pass", image_config=cfg)
        assert sb.image_config is cfg

    def test_packages_and_system_packages(self):
        sb = create(
            name="sb",
            code="pass",
            packages=["numpy", "pandas"],
            system_packages=["git"],
        )
        assert sb.packages == ["numpy", "pandas"]
        assert sb.system_packages == ["git"]

    def test_env_vars(self):
        sb = create(name="sb", code="pass", env_vars={"FOO": "bar"})
        assert sb.env_vars == {"FOO": "bar"}

    def test_cache_options(self):
        for val in ("auto", "override", "disable"):
            sb = create(name="sb", code="pass", cache=val)
            assert sb.cache == val


# ---------------------------------------------------------------------------
# _Sandbox._default_image_name
# ---------------------------------------------------------------------------


class TestDefaultImageName:
    def test_includes_name(self):
        sb = create(name="my-task", code="pass")
        assert sb._default_image_name().startswith("my-task-")

    def test_falls_back_to_sandbox(self):
        sb = create(code="pass")
        assert sb._default_image_name().startswith("sandbox-")

    def test_hash_changes_with_packages(self):
        sb1 = create(name="t", packages=["numpy"])
        sb2 = create(name="t", packages=["pandas"])
        assert sb1._default_image_name() != sb2._default_image_name()

    def test_hash_stable_for_same_packages(self):
        sb1 = create(name="t", packages=["numpy", "pandas"])
        sb2 = create(name="t", packages=["pandas", "numpy"])
        # sorted before hashing — order shouldn't matter
        assert sb1._default_image_name() == sb2._default_image_name()

    def test_hash_length(self):
        sb = create(name="test", code="pass")
        name = sb._default_image_name()
        # format: "{name}-{12-char-hash}"
        parts = name.split("-")
        assert len(parts[-1]) == 12


# ---------------------------------------------------------------------------
# _Sandbox._create_image_spec
# ---------------------------------------------------------------------------


class TestCreateImageSpec:
    def test_returns_flyte_image(self):
        sb = create(name="sb", code="pass")
        img = sb._create_image_spec()
        assert isinstance(img, flyte.Image)

    def test_gcc_always_present(self):
        sb = create(name="sb", code="pass", system_packages=[])
        img = sb._create_image_spec()
        # gcc is injected if not explicitly listed
        cmds = " ".join(str(img) for img in [img])
        # just assert it doesn't raise and returns an image
        assert img is not None
        assert "gcc" in cmds

    def test_gcc_not_duplicated(self):
        sb = create(name="sb", code="pass", system_packages=["gcc"])
        # should not raise; gcc is in the list, so it is not added twice
        img = sb._create_image_spec()
        assert img is not None

    def test_packages_applied(self):
        sb = create(name="sb", code="pass", packages=["requests"])
        img = sb._create_image_spec()
        assert img is not None

    def test_image_name_override(self):
        sb = create(name="sb", code="pass", image_name="custom-image-name")
        img = sb._create_image_spec()
        assert img is not None


# ---------------------------------------------------------------------------
# _Sandbox._make_container_task — code mode
# ---------------------------------------------------------------------------


class TestMakeContainerTaskCodeMode:
    def _make(self, **kwargs) -> ContainerTask:
        sb = create(name="test-sb", code="print('hello')", **kwargs)
        return sb._make_container_task(image="myimage:latest", task_name="test-sb")

    def test_returns_container_task(self):
        task = self._make()
        assert isinstance(task, ContainerTask)

    def test_task_name(self):
        task = self._make()
        assert task.name == "test-sb"

    def test_includes_script_in_inputs(self):
        task = self._make(inputs={"x": int})
        assert "_script" in task._inputs
        assert task._inputs["_script"] is File

    def test_user_inputs_preserved(self):
        task = self._make(inputs={"x": int, "y": str})
        assert "x" in task._inputs
        assert "y" in task._inputs

    def test_file_input_uses_path_argument(self):
        task = self._make(inputs={"data": File})
        # File inputs should appear in arguments list as /var/inputs/<name>
        assert any("/var/inputs/data" in arg for arg in (task._args or []))

    def test_scalar_input_uses_template_syntax(self):
        task = self._make(inputs={"count": int})
        bash_cmd = task._cmd[-1]
        assert "{{.inputs.count}}" in bash_cmd

    def test_outputs_passed_through(self):
        task = self._make(outputs={"result": str, "score": float})
        assert task._outputs == {"result": str, "score": float}

    def test_resources_default(self):
        task = self._make()
        assert task.resources is not None

    def test_resources_override(self):
        res = flyte.Resources(cpu=4, memory="8Gi")
        task = self._make(resources=res)
        assert task.resources is res


# ---------------------------------------------------------------------------
# _Sandbox._generate_auto_script
# ---------------------------------------------------------------------------


class TestGenerateAutoScript:
    def _script(self, code: str, inputs=None, outputs=None) -> str:
        sb = create(name="t", code=code, inputs=inputs or {}, outputs=outputs or {})
        return sb._generate_auto_script()

    def test_no_inputs_no_outputs(self):
        script = self._script("x = 1")
        assert "x = 1" in script
        assert "_parser" not in script
        assert "_out_" not in script

    def test_int_input_generates_argparse(self):
        script = self._script("result = n * 2", inputs={"n": int})
        assert "_parser.add_argument('--n', type=int)" in script
        assert "n = _args.n" in script

    def test_float_input(self):
        script = self._script("y = x + 1.0", inputs={"x": float})
        assert "_parser.add_argument('--x', type=float)" in script

    def test_str_input(self):
        script = self._script("out = msg.upper()", inputs={"msg": str})
        assert "_parser.add_argument('--msg', type=str)" in script

    def test_bool_input_uses_lambda(self):
        script = self._script("pass", inputs={"flag": bool})
        assert "lambda" in script
        assert "'--flag'" in script

    def test_file_input_as_str(self):
        script = self._script("pass", inputs={"data": File})
        assert "_parser.add_argument('--data', type=str)" in script
        assert "data = _args.data" in script

    def test_int_output_epilogue(self):
        script = self._script("result = 42", outputs={"result": int})
        assert "(_out_ / 'result').write_text(str(result))" in script
        assert "_out_ = _pl_.Path('/var/outputs')" in script

    def test_datetime_output_uses_isoformat(self):
        import datetime

        script = self._script("ts = datetime.datetime.now()", outputs={"ts": datetime.datetime})
        assert ".isoformat()" in script

    def test_file_output_not_in_epilogue(self):
        script = self._script("pass", outputs={"out_file": File})
        assert "_out_" not in script  # no epilogue for File outputs

    def test_user_code_is_verbatim(self):
        user_code = "result = x ** 2 + y ** 2"
        script = self._script(user_code, inputs={"x": int, "y": int}, outputs={"result": int})
        assert user_code in script

    def test_user_code_is_dedented(self):
        indented_code = """
            result = n * 2
        """
        script = self._script(indented_code, inputs={"n": int}, outputs={"result": int})
        assert "    result" not in script
        assert "result = n * 2" in script

    def test_generated_script_is_valid_python(self):
        import ast

        script = self._script(
            "result = x * 2",
            inputs={"x": int, "data": File},
            outputs={"result": int},
        )
        ast.parse(script)  # raises SyntaxError if invalid


# ---------------------------------------------------------------------------
# auto_io — make_container_task differences
# ---------------------------------------------------------------------------


class TestAutoIO:
    def _make_verbatim(self, **kwargs) -> ContainerTask:
        sb = create(name="t", code="import sys; print(sys.argv)", auto_io=False, **kwargs)
        return sb._make_container_task("img:latest", "t")

    def _make_auto(self, **kwargs) -> ContainerTask:
        # auto_io=True is the default
        sb = create(name="t", code="result = 1", **kwargs)
        return sb._make_container_task("img:latest", "t")

    def test_verbatim_forwards_cli_args(self):
        # Verbatim mode now forwards CLI args so scripts can use argparse
        task = self._make_verbatim(inputs={"x": int})
        bash_cmd = task._cmd[-1]
        assert "{{.inputs.x}}" in bash_cmd

    def test_verbatim_bash_cmd_is_simple(self):
        task = self._make_verbatim()
        bash_cmd = task._cmd[-1]
        assert "python $1" in bash_cmd

    def test_auto_mode_injects_template_for_scalar(self):
        task = self._make_auto(inputs={"x": int})
        bash_cmd = task._cmd[-1]
        assert "{{.inputs.x}}" in bash_cmd

    def test_auto_mode_injects_path_for_file(self):
        task = self._make_auto(inputs={"data": File})
        bash_cmd = task._cmd[-1]
        assert "--data" in bash_cmd


# ---------------------------------------------------------------------------
# _Sandbox.run — script content written differs by mode
# ---------------------------------------------------------------------------


class TestRunScriptContent:
    def test_auto_mode_generates_preamble(self):
        # auto_io=True is the default
        sb = create(name="t", code="result = n * 2", inputs={"n": int}, outputs={"result": int})
        generated = sb._generate_auto_script()
        assert "_parser" in generated
        assert "n = _args.n" in generated
        assert "write_text" in generated

    def test_verbatim_uses_code_as_is(self):
        raw = "import sys\nprint(sys.argv)"
        sb = create(name="t", code=raw, auto_io=False)
        assert sb.code == raw
        assert sb.auto_io is False


# ---------------------------------------------------------------------------
# _Sandbox._make_container_task — command mode
# ---------------------------------------------------------------------------


class TestMakeContainerTaskCommandMode:
    def _make(self, **kwargs) -> ContainerTask:
        sb = create(
            name="test-cmd",
            command=["/bin/bash", "-c", "echo hi"],
            **kwargs,
        )
        return sb._make_container_task(image="myimage:latest", task_name="test-cmd")

    def test_returns_container_task(self):
        task = self._make()
        assert isinstance(task, ContainerTask)

    def test_command_passed_through(self):
        task = self._make()
        assert task._cmd == ["/bin/bash", "-c", "echo hi"]

    def test_no_script_in_inputs(self):
        task = self._make(inputs={"x": int})
        assert "_script" not in (task._inputs or {})

    def test_arguments_passed(self):
        sb = create(
            name="cmd-args",
            command=["pytest"],
            arguments=["--tb=short", "/var/inputs/tests.py"],
        )
        task = sb._make_container_task("img:latest", "cmd-args")
        assert task._args == ["--tb=short", "/var/inputs/tests.py"]


# ---------------------------------------------------------------------------
# Supported input/output types
# ---------------------------------------------------------------------------


class TestSupportedTypes:
    """Verify that all documented input/output types are accepted without error."""

    SUPPORTED_SCALAR_TYPES: ClassVar[list[type]] = [
        int,
        float,
        str,
        bool,
        datetime.datetime,
        datetime.timedelta,
    ]
    SUPPORTED_IO_TYPES: ClassVar[list[type]] = [File]

    @pytest.mark.parametrize("t", SUPPORTED_SCALAR_TYPES)
    def test_scalar_input_type_accepted(self, t):
        sb = create(name="t", code="pass", inputs={"val": t}, auto_io=True)
        task = sb._make_container_task("img:latest", "t")
        assert isinstance(task, ContainerTask)

    @pytest.mark.parametrize("t", SUPPORTED_IO_TYPES)
    def test_io_input_type_accepted(self, t):
        sb = create(name="t", code="pass", inputs={"val": t}, auto_io=True)
        task = sb._make_container_task("img:latest", "t")
        assert isinstance(task, ContainerTask)

    @pytest.mark.parametrize("t", SUPPORTED_SCALAR_TYPES)
    def test_scalar_output_type_accepted(self, t):
        sb = create(name="t", code="pass", outputs={"out": t}, auto_io=True)
        task = sb._make_container_task("img:latest", "t")
        assert isinstance(task, ContainerTask)

    @pytest.mark.parametrize("t", SUPPORTED_IO_TYPES)
    def test_io_output_type_accepted(self, t):
        sb = create(name="t", code="pass", outputs={"out": t}, auto_io=True)
        task = sb._make_container_task("img:latest", "t")
        assert isinstance(task, ContainerTask)


# ---------------------------------------------------------------------------
# _Sandbox.as_task
# ---------------------------------------------------------------------------


class TestAsTask:
    """Tests for the public as_task() method."""

    @pytest.fixture()
    def mock_file(self):
        """A mock File object returned by File.from_local."""
        mock = AsyncMock()
        mock.return_value = File(path="s3://bucket/script.py")
        return mock

    def test_as_task_returns_container_task(self, mock_file):
        sb = create(
            name="test-as-task",
            code="result = 1",
            inputs={"x": int},
            outputs={"result": int},
        )
        with patch.object(File, "from_local", mock_file):
            task = sb.as_task(image="myimage:latest")
        assert isinstance(task, ContainerTask)

    def test_as_task_sets_script_default(self, mock_file):
        sb = create(
            name="test-as-task",
            code="result = 1",
            inputs={"x": int},
            outputs={"result": int},
        )
        with patch.object(File, "from_local", mock_file):
            task = sb.as_task(image="myimage:latest")
        # _script should have a File default, not None
        script_type, script_default = task.interface.inputs["_script"]
        assert script_type is File
        assert isinstance(script_default, File)

    def test_as_task_script_not_required(self, mock_file):
        sb = create(
            name="test-as-task",
            code="result = 1",
            inputs={"x": int},
            outputs={"result": int},
        )
        with patch.object(File, "from_local", mock_file):
            task = sb.as_task(image="myimage:latest")
        # _script has a default, so it should NOT be in required_inputs
        assert "_script" not in task.interface.required_inputs()

    def test_as_task_user_inputs_still_required(self, mock_file):
        sb = create(
            name="test-as-task",
            code="result = x",
            inputs={"x": int},
            outputs={"result": int},
        )
        with patch.object(File, "from_local", mock_file):
            task = sb.as_task(image="myimage:latest")
        # User-declared inputs should still be present (with None default from ContainerTask)
        assert "x" in task.interface.inputs

    def test_as_task_preserves_outputs(self, mock_file):
        sb = create(name="test-as-task", code="result = 1", outputs={"result": int})
        with patch.object(File, "from_local", mock_file):
            task = sb.as_task(image="myimage:latest")
        assert "result" in task.interface.outputs

    def test_as_task_does_not_set_parent_env(self, mock_file):
        sb = create(name="test-as-task", code="pass")
        with patch.object(File, "from_local", mock_file):
            task = sb.as_task(image="myimage:latest")
        # as_task() returns a clean task with no parent_env,
        # so it can be passed to TaskEnvironment.from_task() for deployment.
        assert task.parent_env is None

    def test_as_task_command_mode_no_script_default(self):
        sb = create(name="test-cmd", command=["/bin/bash", "-c", "echo hi"])
        task = sb.as_task(image="myimage:latest")
        # Command mode has no _script input at all
        assert "_script" not in task.interface.inputs

    def test_as_task_auto_io_generates_preamble(self, mock_file):
        sb = create(
            name="test-as-task",
            code="result = n * 2",
            inputs={"n": int},
            outputs={"result": int},
        )
        with patch.object(File, "from_local", mock_file):
            sb.as_task(image="myimage:latest")
        # Verify script was uploaded (from_local was called)
        mock_file.assert_called_once()

    def test_as_task_verbatim_mode(self, mock_file):
        raw_code = "import sys; print(sys.argv)"
        sb = create(name="test-as-task", code=raw_code, auto_io=False)
        with patch.object(File, "from_local", mock_file):
            task = sb.as_task(image="myimage:latest")
        assert isinstance(task, ContainerTask)
        mock_file.assert_called_once()

    def test_as_task_interface_is_native_interface(self, mock_file):
        sb = create(
            name="test-as-task",
            code="result = 1",
            inputs={"x": int},
            outputs={"result": int},
        )
        with patch.object(File, "from_local", mock_file):
            task = sb.as_task(image="myimage:latest")
        assert isinstance(task.interface, NativeInterface)

    def test_as_task_with_pre_built_image_skips_build(self, mock_file):
        sb = create(name="test-as-task", code="pass", image="prebuilt:latest")
        with patch.object(File, "from_local", mock_file):
            task = sb.as_task()
        # Should use the pre-built image, not trigger a build
        assert isinstance(task, ContainerTask)


# ---------------------------------------------------------------------------
# Public API via flyte.sandbox
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_create_accessible_via_flyte_sandbox(self):
        assert hasattr(flyte.sandbox, "create")
        assert callable(flyte.sandbox.create)

    def test_image_config_accessible_via_flyte_sandbox(self):
        assert hasattr(flyte.sandbox, "ImageConfig")

    def test_create_returns_sandbox_instance(self):
        sb = flyte.sandbox.create(name="public-api-test", code="pass")
        assert isinstance(sb, _Sandbox)

    def test_as_task_is_public(self):
        sb = flyte.sandbox.create(name="test", code="pass")
        assert hasattr(sb, "as_task")
        assert callable(sb.as_task)
