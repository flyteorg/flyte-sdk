from flyte.app._parameter import (
    AppEndpoint,
    Parameter,
    RunOutput,
    SerializableParameter,
    SerializableParameterCollection,
)
from flyte.io import Dir, File


def test_serializable_from_string_parameter():
    p = Parameter(name="config", value="config.yaml")
    sp = SerializableParameter.from_parameter(p)
    assert sp.name == "config"
    assert sp.value == "config.yaml"
    assert sp.type == "string"
    assert sp.download is False


def test_serializable_from_file_parameter():
    p = Parameter(name="model", value=File(path="s3://bucket/model.pkl"))
    sp = SerializableParameter.from_parameter(p)
    assert sp.name == "model"
    assert sp.type == "file"
    assert sp.download is False


def test_serializable_from_file_with_mount():
    p = Parameter(name="model", value=File(path="s3://bucket/model.pkl"), mount="/mnt/model")
    sp = SerializableParameter.from_parameter(p)
    assert sp.download is True
    assert sp.dest == "/mnt/model"


def test_serializable_from_dir_parameter():
    p = Parameter(name="data", value=Dir(path="s3://bucket/data/"))
    sp = SerializableParameter.from_parameter(p)
    assert sp.type == "directory"


def test_serializable_from_run_output():
    ro = RunOutput(type="file", run_name="my-run")
    p = Parameter(name="output", value=ro)
    sp = SerializableParameter.from_parameter(p)
    assert sp.type == "file"
    assert "my-run" in sp.value


def test_serializable_from_app_endpoint():
    ae = AppEndpoint(app_name="upstream")
    p = Parameter(name="api", value=ae, env_var="API_URL")
    sp = SerializableParameter.from_parameter(p)
    assert sp.type == "app_endpoint"
    assert sp.env_var == "API_URL"
    assert "upstream" in sp.value
    assert sp.download is False


def test_collection_roundtrip():
    params = [
        Parameter(name="config", value="config.yaml"),
        Parameter(name="model", value=RunOutput(type="file", run_name="run-1")),
        Parameter(name="api", value=AppEndpoint(app_name="upstream")),
    ]
    collection = SerializableParameterCollection.from_parameters(params)
    transport = collection.to_transport
    assert isinstance(transport, str)
    assert len(transport) > 0

    restored = SerializableParameterCollection.from_transport(transport)
    assert len(restored.parameters) == 3
    assert restored.parameters[0].name == "config"
    assert restored.parameters[0].value == "config.yaml"
    assert restored.parameters[1].name == "model"
    assert "run-1" in restored.parameters[1].value
    assert restored.parameters[2].name == "api"
    assert "upstream" in restored.parameters[2].value
