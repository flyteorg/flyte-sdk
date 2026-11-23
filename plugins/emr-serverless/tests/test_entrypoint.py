import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from flyteplugins.emr_serverless import _entrypoint
from flyteplugins.emr_serverless.connector import EMRServerlessConnector

ENTRYPOINT_PATH = Path(_entrypoint.__file__).resolve()


def test_entrypoint_is_the_staged_file() -> None:
    EMRServerlessConnector._entrypoint_bytes_cache = None
    content = ENTRYPOINT_PATH.read_bytes()
    assert EMRServerlessConnector._ENTRYPOINT_PATH == ENTRYPOINT_PATH
    assert EMRServerlessConnector._read_entrypoint_bytes() == content
    assert len(hashlib.sha256(content).hexdigest()[:12]) == 12


def test_option_value() -> None:
    command = ["a0", "--inputs", "s3://bucket/inputs.pb", "--outputs-path", "s3://bucket/a0"]
    assert _entrypoint._option_value(command, "--outputs-path") == "s3://bucket/a0"
    assert _entrypoint._option_value(command, "--missing") is None


def test_main_runs_a0_and_preserves_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    command = ["a0", "--inputs", "s3://bucket/inputs.pb", "--tgz", "s3://bucket/code.tgz", "--dest", "."]
    seen = []
    monkeypatch.setattr(sys, "argv", ["_entrypoint.py", *command])
    monkeypatch.setattr(_entrypoint, "_run_subprocess", lambda value: seen.extend(value) or 0)
    monkeypatch.setattr(_entrypoint, "_has_runtime_error", lambda value: False)
    with pytest.raises(SystemExit) as error:
        _entrypoint.main()
    assert error.value.code == 0
    assert seen == command


def test_main_turns_error_artifact_into_emr_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["_entrypoint.py", "a0", "--outputs-path", "s3://bucket/a0"])
    monkeypatch.setattr(_entrypoint, "_run_subprocess", lambda value: 0)
    monkeypatch.setattr(_entrypoint, "_has_runtime_error", lambda value: True)
    with pytest.raises(SystemExit) as error:
        _entrypoint.main()
    assert error.value.code == 1


def test_main_propagates_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["_entrypoint.py", "a0"])
    monkeypatch.setattr(_entrypoint, "_run_subprocess", lambda value: 17)
    monkeypatch.setattr(_entrypoint, "_has_runtime_error", lambda value: False)
    with pytest.raises(SystemExit) as error:
        _entrypoint.main()
    assert error.value.code == 17


def test_script_rejects_non_a0_command() -> None:
    process = subprocess.run(
        [sys.executable, str(ENTRYPOINT_PATH), "pyflyte-execute"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 2
    assert "Usage" in process.stderr
