import pytest

from flyte.app import AppEndpoint, Parameter, RunOutput
from flyte.io import Dir, File


def test_parameter_string_value():
    p = Parameter(name="config", value="config.yaml")
    assert p.name == "config"
    assert p.value == "config.yaml"
    assert p.env_var is None
    assert p.download is True
    assert p.mount is None


def test_parameter_with_env_var():
    p = Parameter(name="config", value="config.yaml", env_var="CONFIG_PATH")
    assert p.env_var == "CONFIG_PATH"


def test_parameter_invalid_env_var():
    with pytest.raises(ValueError, match="not a valid environment name"):
        Parameter(name="config", value="val", env_var="invalid-name")


def test_parameter_with_download():
    p = Parameter(name="model", value="s3://bucket/model.pkl", download=True)
    assert p.download is True


def test_parameter_with_mount():
    p = Parameter(name="data", value="s3://bucket/data", mount="/mnt/data")
    assert p.mount == "/mnt/data"


def test_parameter_with_ignore_patterns():
    p = Parameter(name="dir", value="s3://bucket/dir", ignore_patterns=["*.tmp", "*.log"])
    assert p.ignore_patterns == ["*.tmp", "*.log"]


def test_parameter_with_file_value():
    f = File(path="s3://bucket/model.bin")
    p = Parameter(name="model", value=f)
    assert isinstance(p.value, File)


def test_parameter_with_dir_value():
    d = Dir(path="s3://bucket/data/")
    p = Parameter(name="data", value=d)
    assert isinstance(p.value, Dir)


def test_parameter_with_run_output():
    ro = RunOutput(type="file", run_name="my-run-123")
    p = Parameter(name="model", value=ro)
    assert isinstance(p.value, RunOutput)


def test_parameter_with_app_endpoint():
    ae = AppEndpoint(app_name="upstream-app")
    p = Parameter(name="api_url", value=ae)
    assert isinstance(p.value, AppEndpoint)


def test_parameter_invalid_value_type():
    with pytest.raises(TypeError, match="Expected value to be of type"):
        Parameter(name="bad", value=12345)


def test_parameter_name_defaults_to_i0():
    p = Parameter(name=None, value="test")
    assert p.name == "i0"


def test_parameter_env_var_underscore_prefix():
    p = Parameter(name="x", value="v", env_var="_PRIVATE")
    assert p.env_var == "_PRIVATE"


def test_parameter_env_var_with_numbers():
    p = Parameter(name="x", value="v", env_var="VAR_123")
    assert p.env_var == "VAR_123"


def test_parameter_none_value():
    p = Parameter(name="data")
    assert p.name == "data"
    assert p.value is None


def test_parameter_none_value_with_mount():
    p = Parameter(name="model", mount="/tmp/model")
    assert p.value is None
    assert p.mount == "/tmp/model"
