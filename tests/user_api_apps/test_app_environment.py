import pytest

from flyte._image import Image
from flyte._resources import Resources
from flyte.app import AppEnvironment, Domain, Link, Parameter, Port, Scaling, Timeouts


def test_app_environment_minimal():
    app = AppEnvironment(name="my-app", image=Image.from_base("python:3.12"))
    assert app.name == "my-app"
    assert isinstance(app.port, Port)
    assert app.port.port == 8080
    assert app.requires_auth is True
    assert app.cluster_pool == "default"
    assert app.type is None


def test_app_environment_port_int_to_port():
    app = AppEnvironment(name="my-app", image="auto", port=3000)
    assert isinstance(app.port, Port)
    assert app.port.port == 3000


def test_app_environment_port_object():
    port = Port(port=8501, name="h2c")
    app = AppEnvironment(name="my-app", image="auto", port=port)
    assert app.port == port


@pytest.mark.parametrize("reserved_port", [8012, 8022, 8112, 9090, 9091])
def test_app_environment_reserved_port(reserved_port):
    with pytest.raises(ValueError):
        AppEnvironment(name="my-app", image="auto", port=reserved_port)


def test_app_environment_valid_names():
    for name in ["my-app", "app123", "a0", "my-app-v2"]:
        app = AppEnvironment(name=name, image="auto")
        assert app.name == name


def test_app_environment_invalid_names():
    for name in ["My-App", "my_app", "-app", "app-", "APP"]:
        with pytest.raises(ValueError, match="must consist of lower case"):
            AppEnvironment(name=name, image="auto")


def test_app_environment_with_scaling():
    scaling = Scaling(replicas=(1, 5), metric=Scaling.Concurrency(val=10))
    app = AppEnvironment(name="scaled-app", image="auto", scaling=scaling)
    assert app.scaling.replicas == (1, 5)
    assert app.scaling.metric.val == 10


def test_app_environment_with_domain():
    domain = Domain(subdomain="my-sub", custom_domain="app.example.com")
    app = AppEnvironment(name="domain-app", image="auto", domain=domain)
    assert app.domain == domain


def test_app_environment_with_links():
    links = [
        Link(path="/docs", title="API Docs"),
        Link(path="/health", title="Health", is_relative=True),
    ]
    app = AppEnvironment(name="linked-app", image="auto", links=links)
    assert len(app.links) == 2


def test_app_environment_with_parameters():
    params = [
        Parameter(name="config", value="config.yaml", env_var="CONFIG"),
        Parameter(name="model", value="s3://bucket/model.bin"),
    ]
    app = AppEnvironment(name="param-app", image="auto", parameters=params)
    assert len(app.parameters) == 2


def test_app_environment_with_resources():
    app = AppEnvironment(
        name="resource-app",
        image="auto",
        resources=Resources(cpu="2", memory="4Gi"),
    )
    assert app.resources.cpu == "2"
    assert app.resources.memory == "4Gi"


def test_app_environment_with_timeouts():
    t = Timeouts(request=60)
    app = AppEnvironment(name="timeout-app", image="auto", timeouts=t)
    assert app.timeouts.request.total_seconds() == 60


def test_app_environment_with_env_vars():
    app = AppEnvironment(name="env-app", image="auto", env_vars={"KEY": "VALUE", "FOO": "BAR"})
    assert app.env_vars == {"KEY": "VALUE", "FOO": "BAR"}


def test_app_environment_with_secrets():
    app = AppEnvironment(name="secret-app", image="auto", secrets="my-secret")
    assert app.secrets == "my-secret"


def test_app_environment_no_auth():
    app = AppEnvironment(name="public-app", image="auto", requires_auth=False)
    assert app.requires_auth is False


def test_app_environment_type_field():
    app = AppEnvironment(name="typed-app", image="auto", type="streamlit")
    assert app.type == "streamlit"


def test_app_environment_args_list():
    app = AppEnvironment(name="args-app", image="auto", args=["--debug", "--port", "8080"])
    assert app.args == ["--debug", "--port", "8080"]


def test_app_environment_args_string():
    app = AppEnvironment(name="args-app", image="auto", args="--debug --port 8080")
    assert app.args == "--debug --port 8080"


def test_app_environment_command_list():
    app = AppEnvironment(name="cmd-app", image="auto", command=["python", "app.py"])
    assert app.command == ["python", "app.py"]


def test_app_environment_command_string():
    app = AppEnvironment(name="cmd-app", image="auto", command="python app.py")
    assert app.command == "python app.py"


def test_app_environment_include():
    app = AppEnvironment(name="include-app", image="auto", include=["app.py", "config/"])
    assert app.include == ["app.py", "config/"]


def test_app_environment_invalid_args_type():
    with pytest.raises(TypeError, match="Expected args to be of type"):
        AppEnvironment(name="bad-args", image="auto", args=123)


def test_app_environment_invalid_command_type():
    with pytest.raises(TypeError, match="Expected command to be of type"):
        AppEnvironment(name="bad-cmd", image="auto", command=123)


def test_app_environment_invalid_scaling_type():
    with pytest.raises(TypeError, match="Expected scaling to be of type"):
        AppEnvironment(name="bad-scaling", image="auto", scaling="invalid")


def test_app_environment_invalid_domain_type():
    with pytest.raises(TypeError, match="Expected domain to be of type"):
        AppEnvironment(name="bad-domain", image="auto", domain="invalid")


def test_app_environment_invalid_links_type():
    with pytest.raises(TypeError, match="Expected links to be of type"):
        AppEnvironment(name="bad-links", image="auto", links=["not-a-link"])


def test_app_environment_invalid_timeouts_type():
    with pytest.raises(TypeError, match="Expected timeouts to be of type"):
        AppEnvironment(name="bad-timeouts", image="auto", timeouts="invalid")


def test_app_environment_get_port():
    app = AppEnvironment(name="port-app", image="auto", port=5000)
    port = app.get_port()
    assert isinstance(port, Port)
    assert port.port == 5000


def test_app_environment_lifecycle_decorators():
    app = AppEnvironment(name="lifecycle-app", image="auto")

    @app.on_startup
    def startup():
        pass

    @app.server
    def serve():
        pass

    @app.on_shutdown
    def shutdown():
        pass

    assert app._on_startup is startup
    assert app._server is serve
    assert app._on_shutdown is shutdown


def test_app_environment_domain_none():
    app = AppEnvironment(name="no-domain", image="auto", domain=None)
    assert app.domain is None


def test_app_environment_cluster_pool():
    app = AppEnvironment(name="pool-app", image="auto", cluster_pool="gpu-pool")
    assert app.cluster_pool == "gpu-pool"


def test_parameters_with_custom_command_string_raises():
    with pytest.raises(ValueError, match="Cannot use 'parameters' with a custom 'command'"):
        AppEnvironment(
            name="bad-combo",
            image="auto",
            parameters=[Parameter(name="config", value="config.yaml")],
            command="python app.py",
        )


def test_parameters_with_custom_command_list_raises():
    with pytest.raises(ValueError, match="Cannot use 'parameters' with a custom 'command'"):
        AppEnvironment(
            name="bad-combo",
            image="auto",
            parameters=[Parameter(name="config", value="config.yaml")],
            command=["python", "app.py"],
        )


def test_parameters_with_fserve_command_string_allowed():
    app = AppEnvironment(
        name="fserve-app",
        image="auto",
        parameters=[Parameter(name="config", value="config.yaml")],
        command="fserve --version v1",
    )
    assert len(app.parameters) == 1


def test_parameters_with_fserve_command_list_allowed():
    app = AppEnvironment(
        name="fserve-app",
        image="auto",
        parameters=[Parameter(name="config", value="config.yaml")],
        command=["fserve", "--version", "v1"],
    )
    assert len(app.parameters) == 1


def test_custom_command_without_parameters_allowed():
    app = AppEnvironment(name="custom-cmd", image="auto", command="python app.py")
    assert app.command == "python app.py"


def test_parameters_without_command_allowed():
    app = AppEnvironment(
        name="params-only",
        image="auto",
        parameters=[Parameter(name="config", value="config.yaml")],
    )
    assert len(app.parameters) == 1
    assert app.command is None
