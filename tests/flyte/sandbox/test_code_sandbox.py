"""Tests for flyte.sandbox.create() and the _Sandbox class."""

import datetime
from typing import ClassVar

import pytest

import flyte
import flyte.sandbox
from flyte.extras._container import ContainerTask
from flyte.io import Dir, File
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
        assert sb.block_network is True
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

    def test_block_network_default_true(self):
        sb = create(name="sb", code="pass")
        assert sb.block_network is True

    def test_block_network_can_be_disabled(self):
        sb = create(name="sb", code="pass", block_network=False)
        assert sb.block_network is False

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

    def test_block_network_passed(self):
        task = self._make(block_network=True)
        assert task._block_network is True

        task_no_block = self._make(block_network=False)
        assert task_no_block._block_network is False

    def test_file_input_uses_path_argument(self):
        task = self._make(inputs={"data": File})
        # File inputs should appear in arguments list as /var/inputs/<name>
        assert any("/var/inputs/data" in arg for arg in (task._args or []))

    def test_dir_input_uses_path_argument(self):
        task = self._make(inputs={"dataset": Dir})
        assert any("/var/inputs/dataset" in arg for arg in (task._args or []))

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

    def test_config_returns_network_none_when_block_network(self):
        from flyte.models import SerializationContext

        task = self._make(block_network=True)
        sctx = SerializationContext(project="p", domain="d", version="v", org="o")
        assert task.config(sctx) == {"network_mode": "none"}

    def test_config_returns_empty_when_network_allowed(self):
        from flyte.models import SerializationContext

        task = self._make(block_network=False)
        sctx = SerializationContext(project="p", domain="d", version="v", org="o")
        assert task.config(sctx) == {}


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

    def test_dir_input_as_str(self):
        script = self._script("pass", inputs={"dataset": Dir})
        assert "_parser.add_argument('--dataset', type=str)" in script

    def test_int_output_epilogue(self):
        script = self._script("result = 42", outputs={"result": int})
        assert "(_out_ / 'result').write_text(str(result))" in script
        assert "_out_.mkdir" in script

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

    def test_verbatim_no_cli_args_in_bash(self):
        task = self._make_verbatim(inputs={"x": int})
        bash_cmd = task._cmd[-1]
        # No template substitution injected
        assert "{{.inputs" not in bash_cmd

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

    def test_block_network_default_true(self):
        task = self._make()
        assert task._block_network is True


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
    SUPPORTED_IO_TYPES: ClassVar[list[type]] = [File, Dir]

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
