import json
import tempfile

import pytest

from flyte.app._parameter import RUNTIME_PARAMETERS_FILE, _load_parameters, get_parameter


def test_get_parameter_reads_from_file(monkeypatch):
    _load_parameters.cache_clear()

    params = {"config": "config.yaml", "model": "s3://bucket/model.bin"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(params, f)
        f.flush()
        monkeypatch.setenv(RUNTIME_PARAMETERS_FILE, f.name)

        result = get_parameter("config")
        assert result == "config.yaml"

        result2 = get_parameter("model")
        assert result2 == "s3://bucket/model.bin"

    _load_parameters.cache_clear()


def test_get_parameter_missing_key(monkeypatch):
    _load_parameters.cache_clear()

    params = {"config": "config.yaml"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(params, f)
        f.flush()
        monkeypatch.setenv(RUNTIME_PARAMETERS_FILE, f.name)

        with pytest.raises(KeyError):
            get_parameter("nonexistent")

    _load_parameters.cache_clear()


def test_get_parameter_no_env_var(monkeypatch):
    _load_parameters.cache_clear()
    monkeypatch.delenv(RUNTIME_PARAMETERS_FILE, raising=False)

    with pytest.raises(ValueError, match="Parameters are not mounted"):
        get_parameter("anything")

    _load_parameters.cache_clear()
