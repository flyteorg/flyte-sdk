import json
import re
import textwrap

import pytest
from click.testing import CliRunner

pytest.importorskip("libcst")

from flyte.cli.main import main

SOURCE = textwrap.dedent(
    """
    from flytekit import task

    @task(retries=2)
    def t(x: int) -> int:
        return x
    """
)

SOURCE_WITH_TODO = SOURCE.replace("@task(retries=2)", '@task(labels={"a": "b"})')


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Run from the temp dir with short relative paths, without ruff formatting."""
    monkeypatch.setattr("flyte._migrate._migrate._ruff_binary", lambda: None)
    monkeypatch.chdir(tmp_path)


def text(result) -> str:
    """CLI output without Rich panel borders and line wrapping."""
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", result.output).split())


def test_writes_suffixed_file_and_leaves_source_untouched(runner, tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text(SOURCE)

    result = runner.invoke(main, ["migrate", source.name])

    assert result.exit_code == 0, result.output
    assert source.read_text() == SOURCE
    output = tmp_path / "pipeline_v2.py"
    assert "@env.task(retries=2)" in output.read_text()
    assert "Migrated" in text(result)
    assert list(tmp_path.glob("*.tmp")) == []
    assert output.stat().st_mode & 0o777 == source.stat().st_mode & 0o777


def test_custom_suffix(runner, tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text(SOURCE)

    result = runner.invoke(main, ["migrate", source.name, "--suffix", "_flyte2"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "pipeline_flyte2.py").exists()


def test_refuses_to_overwrite_without_force(runner, tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text(SOURCE)
    output = tmp_path / "pipeline_v2.py"
    output.write_text("existing")

    result = runner.invoke(main, ["migrate", source.name])
    assert result.exit_code == 1
    assert "--force" in text(result)
    assert output.read_text() == "existing"

    result = runner.invoke(main, ["migrate", source.name, "--force"])
    assert result.exit_code == 0, result.output
    assert "@env.task" in output.read_text()


def test_dry_run_prints_code_and_writes_nothing(runner, tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text(SOURCE)

    result = runner.invoke(main, ["migrate", source.name, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "@env.task(retries=2)" in result.output
    assert not (tmp_path / "pipeline_v2.py").exists()


def test_syntax_error_writes_nothing(runner, tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text("from flytekit import task\ndef (:\n")

    result = runner.invoke(main, ["migrate", source.name])

    assert result.exit_code == 1
    assert "syntax error" in text(result)
    assert not (tmp_path / "pipeline_v2.py").exists()


def test_missing_file(runner, tmp_path):
    result = runner.invoke(main, ["migrate", "missing.py"])
    assert result.exit_code == 1
    assert "File not found" in text(result)


def test_nothing_to_migrate(runner, tmp_path):
    source = tmp_path / "plain.py"
    source.write_text("print('hi')\n")

    result = runner.invoke(main, ["migrate", source.name])

    assert result.exit_code == 0, result.output
    assert "Nothing to migrate" in text(result)
    assert not (tmp_path / "plain_v2.py").exists()


def test_strict_exits_2_when_todos_remain(runner, tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text(SOURCE_WITH_TODO)

    result = runner.invoke(main, ["migrate", source.name, "--strict"])

    assert result.exit_code == 2, result.output
    assert (tmp_path / "pipeline_v2.py").exists()
    assert "labels are set per run in v2" in text(result)


def test_strict_exits_0_without_todos(runner, tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text(SOURCE)

    result = runner.invoke(main, ["migrate", source.name, "--strict"])

    assert result.exit_code == 0, result.output


def test_json_output(runner, tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text(SOURCE_WITH_TODO)

    result = runner.invoke(main, ["--output-format", "json", "migrate", source.name])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["status"] == "converted_with_todos"
    assert report["output"].endswith("pipeline_v2.py")
    assert report["environments"] == ["pipeline_env"]
    assert report["todos"][0]["message"].startswith("labels are set per run")
    assert "code" not in report


def test_unknown_rule(runner, tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text(SOURCE)

    result = runner.invoke(main, ["migrate", source.name, "--rules", "tasks,bogus"])

    assert result.exit_code == 1
    assert "Unknown rule(s): bogus" in text(result)


def test_missing_libcst_prints_install_hint(runner, tmp_path, monkeypatch):
    import builtins

    source = tmp_path / "pipeline.py"
    source.write_text(SOURCE)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "flyte._migrate":
            raise ModuleNotFoundError("No module named 'libcst'", name="libcst")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = runner.invoke(main, ["migrate", source.name])

    assert result.exit_code == 1
    assert 'pip install "flyte[migrate]"' in text(result)
