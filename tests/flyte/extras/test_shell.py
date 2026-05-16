"""Unit tests for flyte.extras.shell — type classification, script rendering,
flag specs, output collector resolution. Container execution is integration-tested
elsewhere."""

from __future__ import annotations

import asyncio
import pathlib
import tempfile
from typing import List, Optional

import pytest

import flyte
from flyte.extras import shell
from flyte.extras.shell import FlagSpec, Glob, Stderr, Stdout
from flyte.extras.shell._render import _DICT_SEP, _render_command
from flyte.extras.shell._runtime import _read_process_result
from flyte.extras.shell._types import _classify_input, _is_list_of, _is_optional, _validate_outputs
from flyte.io import Dir, File

# ---------------------------------------------------------------------------
# Type classification
# ---------------------------------------------------------------------------


class TestClassifyInput:
    def test_file(self):
        assert _classify_input("a", File) == "file"

    def test_dir(self):
        assert _classify_input("d", Dir) == "dir"

    def test_list_file(self):
        assert _classify_input("b", list[File]) == "list_file"

    def test_list_file_typing_form(self):
        assert _classify_input("b", List[File]) == "list_file"

    def test_scalar_int(self):
        assert _classify_input("n", int) == "scalar"

    def test_scalar_float(self):
        assert _classify_input("f", float) == "scalar"

    def test_scalar_str(self):
        assert _classify_input("s", str) == "scalar"

    def test_bool(self):
        assert _classify_input("wa", bool) == "bool"

    def test_optional_file(self):
        assert _classify_input("a", Optional[File]) == "file"

    def test_optional_pep604(self):
        assert _classify_input("a", File | None) == "file"

    def test_optional_list_file(self):
        assert _classify_input("b", list[File] | None) == "list_file"

    def test_unsupported_dict(self):
        with pytest.raises(TypeError, match="Unsupported input type"):
            _classify_input("x", dict)

    def test_unsupported_list_str(self):
        # list[str] not supported in v1 — only list[File].
        with pytest.raises(TypeError):
            _classify_input("xs", list[str])


class TestIsOptional:
    def test_optional_unwraps(self):
        is_opt, inner = _is_optional(Optional[File])
        assert is_opt
        assert inner is File

    def test_pep604_union(self):
        is_opt, inner = _is_optional(File | None)
        assert is_opt
        assert inner is File

    def test_non_optional(self):
        is_opt, inner = _is_optional(File)
        assert not is_opt
        assert inner is File

    def test_two_way_union_not_optional(self):
        # str | int is a union but not Optional.
        is_opt, _ = _is_optional(str | int)
        assert not is_opt


class TestIsListOf:
    def test_match(self):
        assert _is_list_of(list[File], File)

    def test_typing_match(self):
        assert _is_list_of(List[File], File)

    def test_inner_mismatch(self):
        assert not _is_list_of(list[Dir], File)

    def test_not_a_list(self):
        assert not _is_list_of(File, File)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


class TestValidateOutputs:
    def test_glob_ok(self):
        _validate_outputs({"bed": Glob("*.bed")})

    def test_outfile_ok(self):
        _validate_outputs({"r": File})

    def test_outdir_ok(self):
        _validate_outputs({"d": Dir})

    def test_primitive_ok(self):
        _validate_outputs({"count": int, "ratio": float, "name": str, "ok": bool})

    def test_stdout_stderr_ok(self):
        _validate_outputs({"out": Stdout(), "err": Stderr(type=str)})

    def test_bad_string(self):
        with pytest.raises(TypeError, match="expected a bare type"):
            _validate_outputs({"x": "report.html"})

    def test_bad_unrelated_type(self):
        with pytest.raises(TypeError, match="expected a bare type"):
            _validate_outputs({"x": list})


# ---------------------------------------------------------------------------
# FlagSpec
# ---------------------------------------------------------------------------


class TestFlagSpec:
    def test_default_from_name(self):
        s = FlagSpec.coerce("wa", None)
        assert s.flag == "-wa"
        assert s.list_mode == "join"

    def test_string_alias(self):
        s = FlagSpec.coerce("write_a", "--write-a")
        assert s.flag == "--write-a"
        assert s.list_mode == "join"

    def test_tuple_alias_with_mode(self):
        s = FlagSpec.coerce("I", ("-I", "repeat"))
        assert s.flag == "-I"
        assert s.list_mode == "repeat"

    def test_passthrough_flagspec(self):
        original = FlagSpec(flag="INPUT=", separator="")
        s = FlagSpec.coerce("INPUT", original)
        assert s is original

    def test_invalid(self):
        with pytest.raises(TypeError):
            FlagSpec.coerce("x", 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Script rendering
# ---------------------------------------------------------------------------


def _render(script, inputs, outputs=None, flag_aliases=None):
    """Return just the rendered bash body (drops positional_templates)."""
    flag_specs = {n: FlagSpec.coerce(n, (flag_aliases or {}).get(n)) for n in inputs}
    body, _positional = _render_command(
        script=script,
        inputs=inputs,
        outputs=outputs or {},
        flag_specs=flag_specs,
        input_data_dir=pathlib.Path("/var/inputs"),
        output_data_dir=pathlib.Path("/var/outputs"),
    )
    return body


def _render_full(script, inputs, outputs=None, flag_aliases=None):
    """Return ``(body, positional_templates)`` for tests that inspect the argv."""
    flag_specs = {n: FlagSpec.coerce(n, (flag_aliases or {}).get(n)) for n in inputs}
    return _render_command(
        script=script,
        inputs=inputs,
        outputs=outputs or {},
        flag_specs=flag_specs,
        input_data_dir=pathlib.Path("/var/inputs"),
        output_data_dir=pathlib.Path("/var/outputs"),
    )


class TestRenderCommand:
    def test_file_input_to_path(self):
        out = _render("cat {inputs.a}", {"a": File})
        assert "/var/inputs/a" in out

    def test_dir_input_to_path(self):
        out = _render("ls {inputs.d}", {"d": Dir})
        assert "/var/inputs/d" in out

    def test_list_file_to_glob(self):
        out = _render("cat {inputs.b}", {"b": list[File]})
        assert "/var/inputs/b/*" in out

    def test_scalar_goes_to_positional_arg(self):
        # Scalar values now live in bash positional args ($1, $2, …) so values
        # with single quotes / tabs / specials survive without escaping.
        body, positional = _render_full("echo {inputs.x}", {"x": int})
        assert positional == ["{{.inputs.x}}"]
        # The body binds positional $1 to _VAL_x and references it quoted.
        assert '_VAL_x="${1}"' in body
        assert '"${_VAL_x}"' in body
        # No propeller template appears inside the body string itself.
        assert "{{.inputs.x}}" not in body

    def test_scalar_referenced_as_quoted_token(self):
        body, _ = _render_full("echo {inputs.s}", {"s": str})
        # Reference is quoted so spaces / tabs in the value stay one bash token.
        assert '"${_VAL_s}"' in body

    def test_bool_flag_renders_conditional(self):
        out = _render("foo {flags.wa}", {"wa": bool})
        assert "if [" in out
        assert "_FLAG_wa" in out
        assert "-wa" in out

    def test_flag_alias_overrides_default(self):
        out = _render(
            "foo {flags.write_a}",
            {"write_a": bool},
            flag_aliases={"write_a": "--write-a"},
        )
        assert "--write-a" in out
        assert "-write_a" not in out  # default not used

    def test_outfile_reference(self):
        out = _render(
            "tool > {outputs.bed}",
            {},
            outputs={"bed": File},
        )
        # {outputs.bed} now always renders to /var/outputs/<output_name>.
        assert "/var/outputs/bed" in out

    def test_glob_reference_resolves_to_dir(self):
        # Glob's directory IS the canonical /var/outputs/<name> — the user
        # writes files into it; CoPilot reads it as a Dir.
        out = _render(
            "split -l 5 in.txt {outputs.chunks}/chunk_",
            {},
            outputs={"chunks": Glob("chunk_*")},
        )
        assert "/var/outputs/chunks" in out

    def test_outscalar_reference_resolves_to_path(self):
        out = _render(
            "wc -l < /var/inputs/a > {outputs.count}",
            {"a": File},
            outputs={"count": int},
        )
        assert "/var/outputs/count" in out

    def test_stdout_output_reference_rejected(self):
        # Stdout / Stderr are wrapper-managed; referencing them is an error.
        with pytest.raises(KeyError, match="Stdout/Stderr"):
            _render(
                "echo hello > {outputs.log}",
                {},
                outputs={"log": Stdout()},
            )

    def test_unknown_input_raises(self):
        with pytest.raises(KeyError, match="not declared in inputs"):
            _render("foo {inputs.missing}", {"a": File})

    def test_typo_input_namespace(self):
        # `{input.x}` (singular) should be flagged as unrecognized.
        with pytest.raises(ValueError, match="Unrecognized placeholder"):
            _render("foo {input.x}", {"x": int})

    def test_list_file_flag_join(self):
        out = _render("foo {flags.b}", {"b": list[File]})
        assert "/var/inputs/b/*" in out
        assert "-b" in out

    def test_list_file_flag_repeat(self):
        out = _render(
            "foo {flags.I}",
            {"I": list[File]},
            flag_aliases={"I": ("-I", "repeat")},
        )
        # Repeat mode uses a bash for-loop.
        assert "for _f" in out
        assert "-I" in out

    # ---- optional File / Dir flags emit conditionally ----

    def test_required_file_flag_unconditional(self):
        # Non-optional File flag is hardcoded — the caller is contractually
        # required to supply it, so no runtime existence check is needed.
        out = _render("tool {flags.ref}", {"ref": File})
        assert "_FLAG_ref=" in out
        assert "/var/inputs/ref" in out
        # No conditional guarding the assignment.
        assert "if [ -e " not in out

    def test_optional_file_flag_guarded(self):
        # Optional File flag must be guarded — if the caller didn't supply
        # the file, /var/inputs/<name> won't exist and the tool would fail
        # trying to open it.
        out = _render("tool {flags.sites}", {"sites": File | None})
        assert "if [ -e /var/inputs/sites ]" in out
        assert "_FLAG_sites=-sites /var/inputs/sites" in out or "_FLAG_sites='-sites /var/inputs/sites'" in out
        assert '_FLAG_sites=""' in out  # else branch

    def test_optional_dir_flag_guarded(self):
        # Same conditional emission for optional Dir flags.
        out = _render("tool {flags.cache}", {"cache": Dir | None})
        assert "if [ -e /var/inputs/cache ]" in out
        assert '_FLAG_cache=""' in out

    def test_required_dir_flag_unconditional(self):
        out = _render("tool {flags.workdir}", {"workdir": Dir})
        assert "/var/inputs/workdir" in out
        assert "if [ -e " not in out

    def test_positional_index_braced_for_two_digit_indices(self):
        # Regression: bash parses `$10` as `$1` followed by literal `"0"`,
        # silently binding 10+-indexed inputs to the wrong values. Indices
        # must always be braced as `${10}` for correctness, regardless of
        # whether they happen to be single-digit.
        inputs = {f"x{i}": str for i in range(15)}
        out = _render(" ".join(f"{{inputs.{n}}}" for n in inputs), inputs)
        # Index 1 — must still work (the fix uses braces uniformly).
        assert '_VAL_x0="${1}"' in out
        # Index 10 — this is where the bug bit. Must be braced.
        assert '_VAL_x9="${10}"' in out
        # And the bare two-digit form must NOT appear anywhere.
        assert '"$10"' not in out
        assert '"$15"' not in out


# ---------------------------------------------------------------------------
# Output collector resolution
# ---------------------------------------------------------------------------


@pytest.fixture
def flyte_initialized():
    flyte.init()
    yield


@pytest.fixture
def output_dir(tmp_path):
    """A temp dir simulating /var/outputs after a container run."""
    (tmp_path / "out_a.bed").write_text("track1\n")
    (tmp_path / "out_b.bed").write_text("track2\n")
    (tmp_path / "report.html").write_text("<html/>")
    (tmp_path / "stats").mkdir()
    (tmp_path / "stats" / "summary.txt").write_text("ok")
    (tmp_path / "count").write_text("42\n")
    (tmp_path / "ratio.txt").write_text("0.875")
    (tmp_path / "label").write_text("ok\n")
    (tmp_path / "ok_flag").write_text("true\n")
    (tmp_path / "_returncode").write_text("0\n")
    (tmp_path / "_stdout").write_text("hello stdout\n")
    (tmp_path / "_stderr").write_text("hello stderr\n")
    return tmp_path


class TestValidateOutputsExtra:
    def test_unsupported_bare_type_rejected(self):
        # list is not a supported bare output type.
        with pytest.raises(TypeError, match="expected a bare type"):
            _validate_outputs({"x": list})


class TestBuildCommandOutputPlumbing:
    """The wrapper writes outputs directly to ``/var/outputs/<name>`` —
    no settle / move / copy step. The user's script references outputs
    via ``{outputs.<name>}`` which renders to the canonical path.

    For Stdout/Stderr outputs the wrapper redirects the script's stream
    straight to the canonical path (skipping a copy). For OutDir/Glob,
    the directory is pre-created so the script can write into it.
    """

    def _body(self, **kw) -> str:
        return shell.create(name="t", image="debian:12-slim", **kw)._build_command()[2]

    def test_outfile_canonical_path_in_body(self):
        body = self._body(
            outputs={"bed": File},
            script="cat > {outputs.bed}",
        )
        # Script references the canonical path; no settle tail needed.
        assert "/var/outputs/bed" in body
        assert " mv " not in body and " cp " not in body

    def test_outdir_dir_pre_created(self):
        body = self._body(
            outputs={"stats": Dir},
            script="echo > {outputs.stats}/summary.txt",
        )
        assert "mkdir -p /var/outputs/stats" in body
        assert "/var/outputs/stats" in body
        assert " mv " not in body and " cp " not in body

    def test_glob_dir_pre_created(self):
        body = self._body(
            outputs={"chunks": Glob("chunk_*")},
            script="split -l 5 in.txt {outputs.chunks}/chunk_",
        )
        assert "mkdir -p /var/outputs/chunks" in body
        # No settle / find / mv — the script writes directly into the dir.
        assert " find " not in body
        assert " mv " not in body

    def test_outscalar_canonical_path(self):
        body = self._body(
            outputs={"count": int},
            script="wc -l < x > {outputs.count}",
        )
        assert "/var/outputs/count" in body
        assert " mv " not in body and " cp " not in body

    def test_stdout_redirects_directly_to_canonical(self):
        body = self._body(
            outputs={"log": Stdout()},
            script="echo hello",
        )
        # stdout target is /var/outputs/log; no intermediate _stdout, no cp.
        assert "> /var/outputs/log" in body
        assert "> /var/outputs/_stdout" not in body
        assert " cp " not in body

    def test_stderr_redirects_directly_to_canonical(self):
        body = self._body(
            outputs={"err": Stderr(type=str)},
            script="echo oops >&2",
        )
        assert "2> /var/outputs/err" in body
        assert "2> /var/outputs/_stderr" not in body

    def test_diagnostic_stdout_stderr_when_no_declared(self):
        # No Stdout/Stderr outputs declared → wrapper still tees to the
        # diagnostic files for error reporting.
        body = self._body(
            outputs={"bed": File},
            script="cat > {outputs.bed}",
        )
        assert "/var/outputs/_stdout" in body
        assert "/var/outputs/_stderr" in body

    def test_stdout_output_referenced_in_script_rejected(self):
        # Stdout/Stderr are wrapper-managed — referencing via {outputs.X}
        # is a usage error.
        with pytest.raises(KeyError, match="Stdout/Stderr"):
            self._body(
                outputs={"out": Stdout()},
                script="echo hello > {outputs.out}",
            )


class TestContainerOutputsWireMapping:
    """The shell layer maps collector kinds to wire types CoPilot understands."""

    def test_glob_maps_to_dir(self):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={"bed": Glob("*.bed")},
            script="true",
        )
        assert task._container_outputs() == {"bed": Dir}

    def test_outfile_maps_to_file(self):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={"r": File},
            script="true",
        )
        assert task._container_outputs() == {"r": File}

    def test_outdir_maps_to_dir(self):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={"d": Dir},
            script="true",
        )
        assert task._container_outputs() == {"d": Dir}

    def test_outscalar_maps_to_declared_type(self):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={"n": int, "s": str},
            script="true",
        )
        assert task._container_outputs() == {"n": int, "s": str}

    def test_stdout_stderr_map_to_declared_type(self):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={
                "stdout_f": Stdout(),
                "stderr_int": Stderr(type=int),
            },
            script="true",
        )
        assert task._container_outputs() == {"stdout_f": File, "stderr_int": int}


class TestProcessResult:
    def test_reads_streams_and_returncode(self, output_dir):
        pr = asyncio.run(_read_process_result(output_dir))
        assert pr.returncode == 0
        assert pr.stdout == "hello stdout\n"
        assert pr.stderr == "hello stderr\n"

    def test_missing_returncode_yields_minus_one(self, tmp_path):
        pr = asyncio.run(_read_process_result(tmp_path))
        assert pr.returncode == -1
        assert pr.stdout == ""
        assert pr.stderr == ""


# ---------------------------------------------------------------------------
# create() — validation and shape
# ---------------------------------------------------------------------------


class TestCreate:
    def test_minimal(self):
        task = shell.create(
            name="echo",
            image="alpine:3.18",
            inputs={"x": str},
            outputs={"out": File},
            script="echo {inputs.x} > out.txt",
        )
        assert task.name == "echo"
        assert "x" in task.inputs

    def test_unsupported_input_type_rejected(self):
        with pytest.raises(TypeError, match="Unsupported input type"):
            shell.create(
                name="bad",
                image="alpine:3.18",
                inputs={"x": dict},
                outputs={"o": File},
                script="true",
            )

    def test_bad_output_collector(self):
        with pytest.raises(TypeError, match="expected a bare type"):
            shell.create(
                name="bad",
                image="alpine:3.18",
                outputs={"o": "wrong-type-here"},  # type: ignore[dict-item]
                script="true",
            )

    def test_flag_aliases_must_match_inputs(self):
        with pytest.raises(KeyError, match="not declared in inputs"):
            shell.create(
                name="bad",
                image="alpine:3.18",
                inputs={"a": File},
                outputs={"o": File},
                script="true",
                flag_aliases={"missing": "-m"},
            )

    # ---- case-preserving helper names ----

    def test_case_distinct_inputs_are_allowed(self):
        shell.create(
            name="ok_case",
            image="alpine:3.18",
            inputs={"c": bool, "C": bool},
            outputs={"o": File},
            script="true",
        )

    def test_mixed_case_distinct_inputs_are_allowed(self):
        shell.create(
            name="ok_mixed_case",
            image="alpine:3.18",
            inputs={"my_flag": bool, "My_Flag": bool},
            outputs={"o": File},
            script="true",
        )

    def test_helper_names_preserve_input_case(self):
        body = shell.create(
            name="case_helpers",
            image="alpine:3.18",
            inputs={"c": bool, "C": bool},
            outputs={"o": File},
            script="tool {flags.c} {flags.C} > {outputs.o}",
        )._build_command()[2]
        assert "_FLAG_c" in body
        assert "_FLAG_C" in body

    def test_distinct_case_sensitive_names_still_validate(self):
        shell.create(
            name="ok",
            image="alpine:3.18",
            inputs={"a": bool, "b": bool, "ab": bool},
            outputs={"o": File},
            script="true",
        )

    def test_full_bedtools_shape_validates(self):
        # End-to-end create() with the full bedtools example shape — no exec.
        task = shell.create(
            name="bedtools_intersect",
            image="quay.io/biocontainers/bedtools:2.31.1--hf5e1c6e_0",
            inputs={
                "a": File,
                "b": list[File],
                "wa": bool,
                "loj": bool,
                "f": float,
                "names": list[File] | None,
            },
            outputs={"bed": Glob("*.bed")},
            flag_aliases={"names": "-names"},
            script=r"""
                bedtools intersect {flags.wa} {flags.loj} \
                    -a {inputs.a} \
                    -b {inputs.b} \
                    -f {inputs.f} \
                    {flags.names} \
                    > out.bed
            """,
        )
        cmd = task._build_command()
        assert cmd[0] == "/bin/bash"
        assert cmd[1] == "-c"
        body = cmd[2]
        assert "/var/inputs/a" in body
        assert "/var/inputs/b/*" in body
        assert "_FLAG_wa" in body
        assert "_FLAG_loj" in body
        assert "_FLAG_names" in body
        assert "/var/outputs/_returncode" in body

    def test_debug_mode_emits_script_dump(self):
        task = shell.create(
            name="dbg",
            image="alpine:3.18",
            inputs={"x": str},
            outputs={"o": File},
            script="echo {inputs.x} > {outputs.o}",
            debug=True,
        )
        body = task._build_command()[2]
        assert "rendered script" in body
        assert "cat <<'_EOF_' >&2" in body
        assert '( echo "${_VAL_x}" > /var/outputs/o' not in body

    def test_debug_mode_dump_flows_through_declared_stderr(self):
        task = shell.create(
            name="dbg_err",
            image="alpine:3.18",
            inputs={"x": str},
            outputs={"out": Stdout(type=str), "err": Stderr(type=str)},
            script='echo "running: {inputs.x}"',
            debug=True,
        )
        body = task._build_command()[2]
        assert "> /var/outputs/out 2> /var/outputs/err" in body
        assert 'echo "--- shell task: rendered script ---" >&2' in body
        assert "cat <<'_EOF_' >&2" in body


class TestImageAcceptance:
    def test_string_image_passes_through(self):
        task = shell.create(
            name="t",
            image="quay.io/biocontainers/bedtools:2.31.1--hf5e1c6e_0",
            outputs={"o": File},
            script="true",
        )
        # ContainerTask handles the str → Image conversion internally.
        assert task.image == "quay.io/biocontainers/bedtools:2.31.1--hf5e1c6e_0"

    def test_flyte_image_instance_accepted(self):
        """flyte.Image is accepted; shell builds it lazily on first call."""
        img = flyte.Image.from_debian_base().with_pip_packages("requests")
        task = shell.create(
            name="t",
            image=img,
            outputs={"o": File},
            script="true",
        )
        assert task.image is img  # stored as-is until build

    def test_invalid_image_type_rejected(self):
        with pytest.raises(TypeError, match="image must be"):
            shell.create(
                name="t",
                image=42,  # type: ignore[arg-type]
                outputs={"o": File},
                script="true",
            )


class TestEnv:
    def test_env_is_taskenvironment_wrapping_task(self):
        task = shell.create(
            name="bedtools_intersect",
            image="quay.io/biocontainers/bedtools:2.31.1--hf5e1c6e_0",
            outputs={"o": File},
            script="true",
        )
        env = task.env
        assert isinstance(env, flyte.TaskEnvironment)
        assert env.name == "bedtools_intersect"
        assert "bedtools_intersect" in env.tasks

    def test_env_is_memoised(self):
        task = shell.create(
            name="t",
            image="alpine:3.18",
            outputs={"o": File},
            script="true",
        )
        first = task.env
        second = task.env
        assert first is second  # same instance — env was cached

    def test_env_carries_string_image(self):
        """``task.env`` exposes the user-supplied image; ``TaskEnvironment``
        wraps strings into ``Image.from_base`` internally."""
        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={"o": File},
            script="true",
        )
        # from_task wraps the string via Image.from_base — uri round-trips.
        assert task.env.image.uri == "debian:12-slim"


# ---------------------------------------------------------------------------
# dict[str, str] input type
# ---------------------------------------------------------------------------


class TestDictInput:
    def test_classify_dict_str_str(self):
        assert _classify_input("opts", dict[str, str]) == "dict_str"

    def test_classify_optional_dict(self):
        assert _classify_input("opts", dict[str, str] | None) == "dict_str"

    def test_dict_other_value_types_rejected(self):
        with pytest.raises(TypeError, match="Unsupported"):
            _classify_input("opts", dict[str, int])

    def test_inputs_ref_renders_array_expansion(self):
        body, positional = _render_full("tool {inputs.opts}", {"opts": dict[str, str]})
        # Dict gets a positional slot, then a decode preamble allocates an array.
        assert positional == ["{{.inputs.opts}}"]
        assert "_ARR_opts=" in body
        assert "IFS=" in body
        assert '"${_ARR_opts[@]}"' in body

    def test_flags_pairs_mode_default(self):
        body, _ = _render_full(
            "tool {flags.opts}",
            {"opts": dict[str, str]},
        )
        # Default mode is pairs — keys/values become separate argv tokens.
        assert "_FLAG_opts=" in body
        assert '"${_FLAG_opts[@]}"' in body

    def test_flags_equals_mode(self):
        body, _ = _render_full(
            "tool {flags.input}",
            {"input": dict[str, str]},
            flag_aliases={"input": ("INPUT=", "equals")},
        )
        # equals mode combines into key=value tokens with prefix
        assert "+=" in body
        assert "INPUT=" in body

    def test_dict_value_with_single_quote_is_safe(self):
        # The dict goes over the wire as a \x1e-delimited str; single quotes
        # in values never reach a shell-parsing context. We test the encoding
        # side here — the runtime decoding is exercised in TestPrepareKwargs.

        # Just verify the rendering produces no quoted substitution
        body, positional = _render_full(
            "tool {flags.opts}",
            {"opts": dict[str, str]},
        )
        # No '{{.inputs.opts}}' appears inside the body — only the bash decode.
        assert "{{.inputs.opts}}" not in body
        assert positional == ["{{.inputs.opts}}"]


class TestPrepareKwargs:
    def test_dict_packed_to_record_separator_string(self):
        task = shell.create(
            name="t",
            image="alpine:3.18",
            inputs={"opts": dict[str, str]},
            outputs={"o": File},
            script="echo {flags.opts}",
        )
        result = asyncio.run(task._prepare_kwargs({"opts": {"--memory": "4G", "-R": "@RG\tID:x"}}))
        # str on the wire, \x1e separating tokens.
        assert isinstance(result["opts"], str)
        parts = result["opts"].split(_DICT_SEP)
        assert parts == ["--memory", "4G", "-R", "@RG\tID:x"]

    def test_dict_with_record_separator_in_value_rejected(self):
        task = shell.create(
            name="t",
            image="alpine:3.18",
            inputs={"opts": dict[str, str]},
            outputs={"o": File},
            script="echo {flags.opts}",
        )
        with pytest.raises(ValueError, match="record-separator"):
            asyncio.run(task._prepare_kwargs({"opts": {"k": f"a{_DICT_SEP}b"}}))

    def test_optional_dict_default_empty_string(self):
        task = shell.create(
            name="t",
            image="alpine:3.18",
            inputs={"opts": dict[str, str] | None},
            outputs={"o": File},
            script="echo {flags.opts}",
        )
        result = asyncio.run(task._prepare_kwargs({}))
        # Missing optional dict packs to empty string so the bash decode
        # branch becomes a no-op (array stays empty).
        assert result["opts"] == ""


# ---------------------------------------------------------------------------
# Defaults — four-cell matrix of {required, optional} x {has default, none}
# ---------------------------------------------------------------------------


class TestDefaults:
    def _task(self, inputs, defaults=None):
        return shell.create(
            name="t",
            image="alpine:3.18",
            inputs=inputs,
            outputs={"o": File},
            script="true",
            defaults=defaults,
        )

    # ---- four-cell matrix ----

    def test_required_no_default_missing_raises(self):
        task = self._task({"wa": bool})
        with pytest.raises(TypeError, match="Missing required input: 'wa'"):
            asyncio.run(task._prepare_kwargs({}))

    def test_required_with_default_uses_default(self):
        task = self._task({"wa": bool}, defaults={"wa": False})
        result = asyncio.run(task._prepare_kwargs({}))
        # Non-optional bool flows as the native Python bool — ContainerTask
        # lower-cases it to the bash "false" string at template-render time.
        assert result["wa"] is False

    def test_optional_no_default_missing_empty_string(self):
        task = self._task({"wa": bool | None})
        result = asyncio.run(task._prepare_kwargs({}))
        assert result["wa"] == ""

    def test_optional_with_default_uses_default(self):
        task = self._task({"wa": bool | None}, defaults={"wa": True})
        result = asyncio.run(task._prepare_kwargs({}))
        # Optional scalars/bools are wired as str — defaults flow through
        # the same conversion path as caller-supplied values.
        assert result["wa"] == "true"

    # ---- caller-supplied value always wins ----

    def test_caller_value_overrides_default(self):
        task = self._task({"threads": int}, defaults={"threads": 4})
        result = asyncio.run(task._prepare_kwargs({"threads": 8}))
        assert result["threads"] == 8

    def test_caller_none_overrides_default_for_optional(self):
        # Explicit None from caller must win even when a default exists —
        # otherwise there's no way to opt out of an emitted flag.
        task = self._task({"threads": int | None}, defaults={"threads": 4})
        result = asyncio.run(task._prepare_kwargs({"threads": None}))
        assert result["threads"] == ""

    # ---- default values for various kinds ----

    def test_default_for_optional_int(self):
        task = self._task({"n": int | None}, defaults={"n": 42})
        result = asyncio.run(task._prepare_kwargs({}))
        assert result["n"] == "42"

    def test_default_for_optional_float(self):
        task = self._task({"f": float | None}, defaults={"f": 0.5})
        result = asyncio.run(task._prepare_kwargs({}))
        assert result["f"] == "0.5"

    def test_default_for_optional_str(self):
        task = self._task({"s": str | None}, defaults={"s": "hello"})
        result = asyncio.run(task._prepare_kwargs({}))
        assert result["s"] == "hello"

    def test_default_for_optional_dict(self):
        task = self._task({"opts": dict[str, str] | None}, defaults={"opts": {"-k": "v"}})
        result = asyncio.run(task._prepare_kwargs({}))
        # Dict defaults flow through the record-separator packing path.
        assert result["opts"].split(_DICT_SEP) == ["-k", "v"]

    def test_default_for_optional_file(self):
        f = File(path="/tmp/example.txt")
        task = self._task({"src": File | None}, defaults={"src": f})
        result = asyncio.run(task._prepare_kwargs({}))
        assert result["src"] is f

    def test_default_for_optional_dir(self):
        d = Dir(path="/tmp/example_dir")
        task = self._task({"src": Dir | None}, defaults={"src": d})
        result = asyncio.run(task._prepare_kwargs({}))
        assert result["src"] is d

    def test_default_for_optional_list_of_files(self):
        files = [File(path="/tmp/a.txt"), File(path="/tmp/b.txt")]
        task = self._task({"parts": list[File] | None}, defaults={"parts": files})
        result = asyncio.run(task._prepare_kwargs({}))
        assert result["parts"] == files

    # ---- create()-time validation ----

    def test_validate_unknown_key_rejected(self):
        with pytest.raises(KeyError, match="not declared in inputs"):
            self._task({"wa": bool}, defaults={"unknown": True})

    def test_validate_none_default_rejected(self):
        with pytest.raises(ValueError, match="redundant"):
            self._task({"wa": bool | None}, defaults={"wa": None})

    def test_validate_bool_type_mismatch(self):
        with pytest.raises(TypeError, match="expected bool"):
            self._task({"wa": bool}, defaults={"wa": "yes"})

    def test_validate_int_type_mismatch_rejects_bool(self):
        # bool is a subclass of int — reject it for int defaults to keep
        # `True` from quietly meaning `1`.
        with pytest.raises(TypeError, match="expected int"):
            self._task({"n": int}, defaults={"n": True})

    def test_validate_float_accepts_int(self):
        # Lenient: int → float coercion is the obvious user intent.
        task = self._task({"f": float | None}, defaults={"f": 5})
        result = asyncio.run(task._prepare_kwargs({}))
        assert result["f"] == "5"

    def test_validate_str_type_mismatch(self):
        with pytest.raises(TypeError, match="expected str"):
            self._task({"s": str | None}, defaults={"s": 42})

    def test_validate_dict_type_mismatch(self):
        with pytest.raises(TypeError, match="expected dict"):
            self._task({"opts": dict[str, str] | None}, defaults={"opts": "not a dict"})

    def test_validate_dict_non_string_value_rejected(self):
        with pytest.raises(TypeError, match="string keys and values"):
            self._task(
                {"opts": dict[str, str] | None},
                defaults={"opts": {"k": 42}},  # type: ignore[dict-item]
            )

    def test_validate_file_type_mismatch(self):
        with pytest.raises(TypeError, match="expected File"):
            self._task({"src": File | None}, defaults={"src": "/tmp/example.txt"})

    def test_validate_dir_type_mismatch(self):
        with pytest.raises(TypeError, match="expected Dir"):
            self._task({"src": Dir | None}, defaults={"src": "/tmp/example_dir"})

    def test_validate_list_of_files_type_mismatch(self):
        with pytest.raises(TypeError, match=r"expected list\[File\]"):
            self._task({"parts": list[File] | None}, defaults={"parts": "not a list"})

    def test_validate_list_of_files_item_type_mismatch(self):
        with pytest.raises(TypeError, match=r"list\[File\] requires every item to be a File"):
            self._task(
                {"parts": list[File] | None},
                defaults={"parts": [File(path="/tmp/a.txt"), "/tmp/b.txt"]},
            )

    # ---- no defaults parameter at all (backward compat) ----

    def test_no_defaults_param_is_backward_compatible(self):
        task = shell.create(
            name="t",
            image="alpine:3.18",
            inputs={"wa": bool | None},
            outputs={"o": File},
            script="true",
        )
        assert task.defaults == {}
        result = asyncio.run(task._prepare_kwargs({}))
        assert result["wa"] == ""


# ---------------------------------------------------------------------------
# Stdout / Stderr collectors
# ---------------------------------------------------------------------------


class TestStdoutStderrCollectors:
    def test_classify(self):
        # Validation accepts both as outputs.
        _validate_outputs({"out": Stdout(), "err": Stderr()})

    def test_rejects_unsupported_type(self):
        with pytest.raises(TypeError, match=r"Stdout\.type must be"):
            Stdout(type=list)

    def test_cannot_be_referenced_in_script(self):
        # Stream collectors are wrapper-managed; the user must not
        # reference them via {outputs.X} — the wrapper redirects the
        # script's stream directly to the canonical path.
        with pytest.raises(KeyError, match="Stdout/Stderr"):
            _render("tool > {outputs.out}", {}, outputs={"out": Stdout()})

    # Stream resolution is now performed by ContainerTask's default
    # _get_output reading the file at /var/outputs/<output_name>; the
    # wrapper redirects the script's stream straight there. See
    # TestBuildCommandOutputPlumbing and TestContainerOutputsWireMapping
    # for the relevant coverage.


# ---------------------------------------------------------------------------
# Positional-args safety (single quotes, tabs, $, etc.)
# ---------------------------------------------------------------------------


class TestScalarValuesSurviveShellSpecials:
    """Regression cases for the single-quote / shell-special-character class of bugs.

    Scalar values go through bash positional args, never through inline shell
    substitution. The body never contains the literal value — only a
    `"${_VAL_x}"` reference. Propeller substitutes the literal value into the
    argv slot at runtime; bash sees it as a verbatim string.
    """

    def test_literal_value_does_not_appear_in_body(self):
        body, positional = _render_full("echo {inputs.s}", {"s": str})
        # The rendered body must NEVER carry the literal substitution token —
        # otherwise single quotes / dollar signs would break.
        assert "{{.inputs.s}}" not in body
        # The propeller template lives in the positional list, not the body.
        assert "{{.inputs.s}}" in positional

    def test_each_scalar_gets_distinct_positional_slot(self):
        body, positional = _render_full(
            "echo {inputs.a} {inputs.b}",
            {"a": str, "b": int},
        )
        assert positional == ["{{.inputs.a}}", "{{.inputs.b}}"]
        assert '_VAL_a="${1}"' in body
        assert '_VAL_b="${2}"' in body

    def test_same_input_referenced_twice_reuses_slot(self):
        body, positional = _render_full(
            "echo {inputs.x} {inputs.x}",
            {"x": int},
        )
        # x referenced twice — single positional slot.
        assert positional == ["{{.inputs.x}}"]
        assert body.count('_VAL_x="${1}"') == 1

    def test_inputs_and_flags_for_same_var_share_slot(self):
        body, positional = _render_full(
            "tool {flags.f} -override {inputs.f}",
            {"f": str},
        )
        assert positional == ["{{.inputs.f}}"]
        # _VAL_f bound once, used by both the flag setter and the inputs ref.
        assert body.count('_VAL_f="${1}"') == 1


class TestBuildCommandArgvLayout:
    def test_command_appends_positional_templates(self):
        task = shell.create(
            name="t",
            image="alpine:3.18",
            inputs={"x": int, "y": str},
            outputs={"o": File},
            script="tool {inputs.x} {inputs.y} > {outputs.o}",
        )
        cmd = task._build_command()
        # shell -c <body> _shell_task <x_template> <y_template>
        assert cmd[0] == "/bin/bash"
        assert cmd[1] == "-c"
        # cmd[2] is the rendered body
        assert cmd[3] == "_shell_task"
        assert cmd[4:] == ["{{.inputs.x}}", "{{.inputs.y}}"]

    def test_command_omits_positional_for_pure_file_task(self):
        # No scalar/bool/dict inputs -> no positional templates.
        task = shell.create(
            name="t",
            image="alpine:3.18",
            inputs={"a": File},
            outputs={"o": File},
            script="cat {inputs.a} > {outputs.o}",
        )
        cmd = task._build_command()
        assert cmd[3] == "_shell_task"
        assert cmd[4:] == []


# ---------------------------------------------------------------------------
# End-to-end argv composition with shell-special-character values
# ---------------------------------------------------------------------------


class TestEndToEndArgvComposition:
    """Actually run bash with the rendered command and verify argv tokens.

    Uses /bin/bash directly (not docker) — simulates exactly what would
    happen at execute time, with propeller substitutions applied as
    positional bash arguments.
    """

    def _execute_with_values(self, task, **vals):
        """Run the rendered bash body locally; capture the stdout output file.

        If a ``Stdout()`` collector is declared, the wrapper redirects the
        script's stdout straight to ``/var/outputs/<name>``; otherwise it
        goes to ``_stdout`` for diagnostics. We read whichever applies.
        """
        import subprocess

        cmd = task._build_command()
        body = cmd[2]
        # Adjust output capture paths to a tmp dir.
        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="shell-e2e-"))
        body = body.replace("/var/outputs/", str(tmpdir) + "/")
        # Substitute the positional templates with the real values.
        positional = [str(vals[t.removeprefix("{{.inputs.").removesuffix("}}")]) for t in cmd[4:]]
        result = subprocess.run(
            [cmd[0], cmd[1], body, cmd[3], *positional],
            capture_output=True,
            text=True,
            check=False,
        )
        # Find the actual stdout-capture file.
        stdout_name = "_stdout"
        for name, spec in task.outputs.items():
            if isinstance(spec, Stdout):
                stdout_name = name
                break
        stdout_file = tmpdir / stdout_name
        return result.returncode, stdout_file.read_text() if stdout_file.exists() else ""

    def test_scalar_with_single_quote_survives(self):
        task = shell.create(
            name="t",
            image="alpine:3.18",
            inputs={"label": str},
            outputs={"out": Stdout(type=str)},
            script="echo got={inputs.label}",
        )
        rc, out = self._execute_with_values(task, label="it's a test")
        assert rc == 0
        assert "got=it's a test" in out

    def test_scalar_with_dollar_sign_survives(self):
        task = shell.create(
            name="t",
            image="alpine:3.18",
            inputs={"v": str},
            outputs={"out": Stdout(type=str)},
            script="echo got={inputs.v}",
        )
        rc, out = self._execute_with_values(task, v="$PATH literal")
        assert rc == 0
        # Bash never expands $PATH because the value arrives as a verbatim
        # positional arg — never re-parsed in a substitution context.
        assert "got=$PATH literal" in out

    def test_scalar_with_backtick_survives(self):
        task = shell.create(
            name="t",
            image="alpine:3.18",
            inputs={"v": str},
            outputs={"out": Stdout(type=str)},
            script="echo got={inputs.v}",
        )
        rc, out = self._execute_with_values(task, v="`rm -rf /`")
        assert rc == 0
        # Command substitution does not fire — value is a literal string.
        assert "got=`rm -rf /`" in out

    def test_dict_value_with_tab_survives(self):
        task = shell.create(
            name="t",
            image="alpine:3.18",
            inputs={"opts": dict[str, str]},
            outputs={"out": Stdout(type=str)},
            script="echo {flags.opts}",
        )
        # Caller-side packing happens through _prepare_kwargs in real use,
        # but here we already have the encoded str — bypass and pass directly.
        encoded = "-R" + _DICT_SEP + "@RG\tID:x" + _DICT_SEP + "--memory" + _DICT_SEP + "4G"
        rc, out = self._execute_with_values(task, opts=encoded)
        assert rc == 0
        # All four tokens should appear, tab preserved within @RG\tID:x.
        assert "-R" in out
        assert "@RG\tID:x" in out
        assert "--memory" in out
        assert "4G" in out


class TestOutputResolutionErrorDiagnostics:
    """When an output collector fails (file missing at the canonical path),
    the error should include stdout/stderr/returncode for diagnostics."""

    def test_missing_outfile_error_includes_streams(self, tmp_path, flyte_initialized):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            inputs={},
            outputs={"out": File},  # path == name; nothing to mv
            script="true",
        )
        ct = task.as_task()

        # Simulate a failed run: stderr has a real error, no output file at
        # the canonical /var/outputs/out path.
        (tmp_path / "_stdout").write_text("")
        (tmp_path / "_stderr").write_text("head: cannot open '/var/inputs/src' for reading: No such file\n")
        (tmp_path / "_returncode").write_text("1\n")

        with pytest.raises(FileNotFoundError) as ei:
            asyncio.run(ct._get_output(tmp_path))

        msg = str(ei.value)
        assert "returncode=1" in msg
        assert "stderr" in msg
        # The real reason surfaces in the error.
        assert "head: cannot open" in msg

    def test_nonzero_exit_raises_even_when_outputs_exist(self, tmp_path, flyte_initialized):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            inputs={},
            outputs={"count": int},
            script="true",
        )
        ct = task.as_task()

        (tmp_path / "count").write_text("42\n")
        (tmp_path / "_stdout").write_text("partial stdout\n")
        (tmp_path / "_stderr").write_text("tool failed after writing output\n")
        (tmp_path / "_returncode").write_text("1\n")

        with pytest.raises(RuntimeError, match="returncode=1"):
            asyncio.run(ct._get_output(tmp_path))


class TestOptionalScalarWireFormat:
    """Optional scalars / bools marshal as ``str`` on the wire so that
    ``None`` can travel safely (ContainerTask's substitution would otherwise
    stringify None to the literal ``"None"``)."""

    def test_optional_int_wire_type_is_str(self):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            inputs={"n": int | None},
            outputs={"out": File},
            script="echo {flags.n}",
        )
        wired = task._container_inputs()
        assert wired["n"] is str

    def test_required_int_wire_type_is_int(self):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            inputs={"n": int},
            outputs={"out": File},
            script="echo {flags.n}",
        )
        wired = task._container_inputs()
        assert wired["n"] is int

    def test_optional_int_none_packs_to_empty_string(self, flyte_initialized):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            inputs={"n": int | None},
            outputs={"out": File},
            script="echo {flags.n}",
        )
        out = asyncio.run(task._prepare_kwargs({"n": None}))
        assert out["n"] == ""

    def test_optional_int_value_packs_to_str(self, flyte_initialized):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            inputs={"n": int | None},
            outputs={"out": File},
            script="echo {flags.n}",
        )
        out = asyncio.run(task._prepare_kwargs({"n": 8}))
        assert out["n"] == "8"

    def test_optional_bool_packs_to_lowercase_str(self, flyte_initialized):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            inputs={"verbose": bool | None},
            outputs={"out": File},
            script="echo {flags.verbose}",
        )
        for value, expected in [(True, "true"), (False, "false"), (None, "")]:
            out = asyncio.run(task._prepare_kwargs({"verbose": value}))
            assert out["verbose"] == expected, f"value={value!r}"

    def test_missing_optional_packs_to_empty_string(self, flyte_initialized):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            inputs={"n": int | None},
            outputs={"out": File},
            script="echo {flags.n}",
        )
        out = asyncio.run(task._prepare_kwargs({}))
        assert out["n"] == ""

    def test_required_scalar_missing_raises(self, flyte_initialized):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            inputs={"n": int},
            outputs={"out": File},
            script="echo {flags.n}",
        )
        with pytest.raises(TypeError, match="Missing required"):
            asyncio.run(task._prepare_kwargs({}))


class TestLocalLogs:
    def test_default_is_true(self):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={"o": File},
            script="true",
        )
        assert task.local_logs is True
        assert task.as_task().local_logs is True

    def test_can_silence(self):
        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={"o": File},
            script="true",
            local_logs=False,
        )
        assert task.local_logs is False
        assert task.as_task().local_logs is False


class TestResolveImageURI:
    """``shell.create`` accepts both URI strings and ``flyte.Image`` instances.
    String URIs pass through; ``flyte.Image`` is built via the configured
    builder (``cfg.image_builder``) and the resulting URI is what
    ContainerTask receives."""

    def test_string_uri_no_build(self, monkeypatch):
        called = {"n": 0}

        async def fake_build(*a, **k):
            called["n"] += 1

            class R:
                uri = None

            return R()

        monkeypatch.setattr(flyte.build, "aio", fake_build)
        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={"o": File},
            script="true",
        )
        uri = asyncio.run(task._resolve_image_uri())
        assert uri == "debian:12-slim"
        assert called["n"] == 0

    def test_flyte_image_triggers_build(self, monkeypatch):
        called = {"n": 0, "received": None}

        async def fake_build(image, **k):
            called["n"] += 1
            called["received"] = image

            class R:
                uri = "registry.example.com/built:abc123"

            return R()

        monkeypatch.setattr(flyte.build, "aio", fake_build)
        img = flyte.Image.from_debian_base().with_apt_packages("jq")
        task = shell.create(
            name="t",
            image=img,
            outputs={"o": File},
            script="true",
        )
        uri = asyncio.run(task._resolve_image_uri())
        assert uri == "registry.example.com/built:abc123"
        assert called["n"] == 1
        assert called["received"] is img
        # User-facing field unchanged; resolution cached separately.
        assert task.image is img
        assert task._resolved_image_uri == "registry.example.com/built:abc123"

    def test_resolve_is_memoised(self, monkeypatch):
        called = {"n": 0}

        async def fake_build(image, **k):
            called["n"] += 1

            class R:
                uri = "registry.example.com/built:abc"

            return R()

        monkeypatch.setattr(flyte.build, "aio", fake_build)
        task = shell.create(
            name="t",
            image=flyte.Image.from_debian_base().with_apt_packages("jq"),
            outputs={"o": File},
            script="true",
        )
        for _ in range(3):
            asyncio.run(task._resolve_image_uri())
        assert called["n"] == 1  # only first call built; rest cache-hit

    def test_build_returning_no_uri_raises(self, monkeypatch):
        async def fake_build(image, **k):
            class R:
                uri = None

            return R()

        monkeypatch.setattr(flyte.build, "aio", fake_build)
        task = shell.create(
            name="t",
            image=flyte.Image.from_debian_base().with_apt_packages("jq"),
            outputs={"o": File},
            script="true",
        )
        with pytest.raises(RuntimeError, match="returned no URI"):
            asyncio.run(task._resolve_image_uri())


class TestUnpackOutputs:
    """Glob outputs travel over the wire as Dir (a bundle directory built
    by the settle tail); ``_Shell._unpack_outputs`` converts back to
    ``list[File]`` before returning to the caller."""

    def test_glob_unpacks_dir_to_list_of_files(self, tmp_path, flyte_initialized):
        # Simulate the post-execution Dir handle CoPilot/ContainerTask returns:
        # a directory containing the bundled files.
        bundle = tmp_path / "bed_bundle"
        bundle.mkdir()
        (bundle / "a.bed").write_text("track1\n")
        (bundle / "b.bed").write_text("track2\n")
        d = asyncio.run(Dir.from_local(str(bundle)))

        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={"bed": Glob("*.bed")},
            script="true",
        )
        result = asyncio.run(task._unpack_outputs(d))
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(f, File) for f in result)

    def test_glob_unpacks_empty_dir_to_empty_list(self, tmp_path, flyte_initialized):
        bundle = tmp_path / "empty_bundle"
        bundle.mkdir()
        d = asyncio.run(Dir.from_local(str(bundle)))

        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={"bed": Glob("*.bed")},
            script="true",
        )
        result = asyncio.run(task._unpack_outputs(d))
        assert result == []

    def test_non_glob_passes_through(self, tmp_path, flyte_initialized):
        # OutFile output: wire type is File; no unpacking needed.
        f_path = tmp_path / "x.txt"
        f_path.write_text("hello")
        f = asyncio.run(File.from_local(str(f_path)))

        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={"out": File},
            script="true",
        )
        result = asyncio.run(task._unpack_outputs(f))
        assert result is f

    def test_mixed_outputs_unpack_glob_only(self, tmp_path, flyte_initialized):
        bundle = tmp_path / "bed_bundle"
        bundle.mkdir()
        (bundle / "x.bed").write_text("track\n")
        d = asyncio.run(Dir.from_local(str(bundle)))
        f_path = tmp_path / "stats"
        f_path.write_text("ok")
        f = asyncio.run(File.from_local(str(f_path)))

        task = shell.create(
            name="t",
            image="debian:12-slim",
            outputs={
                "bed": Glob("*.bed"),
                "stats": File,
            },
            script="true",
        )
        # Multi-output: task returns a tuple.
        result = asyncio.run(task._unpack_outputs((d, f)))
        assert isinstance(result, tuple)
        assert isinstance(result[0], list) and len(result[0]) == 1
        assert result[1] is f
