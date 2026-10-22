import pytest
from flyteidl2.core import execution_pb2

import flyte
import flyte.errors
from flyte._internal.apple_silicon import _host
from flyte._internal.apple_silicon._token import mint_token, verify_token
from flyte._internal.runtime.convert import convert_error_to_native
from flyte._internal.runtime.resources_serde import get_proto_extended_resources, get_proto_resources


def test_token_matches_the_backend():
    # Same vector as flyteplugins' flytek8s/apple_silicon_test.go (TestMintAppleSiliconToken).
    assert (
        mint_token("s3cret", "abc-123")
        == "as1.abc-123.7b94a54d7edb63bba60a50cb2818a3d36fcd12ed3a85b68650c57e010e03cc50"
    )


def test_token_verification():
    secret = "s3cret" * 8
    token = mint_token(secret, "app-llm-proj-dev")
    assert verify_token(secret, token) == "app-llm-proj-dev"
    assert verify_token("other" * 8, token) is None
    assert verify_token(secret, token.replace("app-llm", "app-xyz")) is None
    assert verify_token(secret, "") is None
    assert verify_token(secret, "as2.a.b") is None
    # A damaged (empty or truncated) secret must not let anyone mint a valid token.
    assert verify_token("", mint_token("", "w")) is None
    assert verify_token("short", mint_token("short", "w")) is None


def test_apple_silicon_accelerator_string():
    device = flyte.Resources(gpu="AppleSilicon:1").get_device()
    assert device == flyte.Device(quantity=1, device_class="APPLE_SILICON", device="AppleSilicon")


def test_apple_silicon_without_a_count():
    # The Mac has one GPU, so the count can be left off.
    assert flyte.Resources(gpu="AppleSilicon").get_device() == flyte.Resources(gpu="AppleSilicon:1").get_device()
    accelerator = get_proto_extended_resources(flyte.Resources(gpu="AppleSilicon")).gpu_accelerator
    assert (accelerator.device, accelerator.device_class) == ("AppleSilicon", 5)
    image = flyte.Image.from_debian_base().with_pip_packages("torch")
    from flyte._internal.runtime.resources_serde import get_apple_silicon_execution_env

    assert (
        get_apple_silicon_execution_env(flyte.Resources(gpu="AppleSilicon"), image, None)["_F_APPLE_SILICON_EXECUTION"]
        == "host"
    )
    # Other accelerators still need their count.
    with pytest.raises(ValueError):
        flyte.Resources(gpu="A100")


def test_apple_silicon_helper():
    assert flyte.APPLE_SILICON() == flyte.Resources(gpu="AppleSilicon").get_device()
    assert flyte.Resources(gpu=flyte.APPLE_SILICON()).get_device() == flyte.APPLE_SILICON()
    accelerator = get_proto_extended_resources(flyte.Resources(gpu=flyte.APPLE_SILICON())).gpu_accelerator
    assert (accelerator.device, accelerator.device_class) == ("AppleSilicon", 5)


def test_apple_silicon_is_the_single_host_gpu():
    with pytest.raises(ValueError):
        flyte.Resources(gpu="AppleSilicon:2")
    with pytest.raises(ValueError, match="quantity must be 1"):
        flyte.Device(quantity=2, device_class="APPLE_SILICON", device="AppleSilicon")


def test_apple_silicon_serializes_as_apple_silicon():
    resources = flyte.Resources(gpu="AppleSilicon", memory="16Gi")
    accelerator = get_proto_extended_resources(resources).gpu_accelerator
    assert accelerator.device == "AppleSilicon"
    assert accelerator.device_class == 5  # APPLE_SILICON
    requests, _ = get_proto_resources(resources).requests, None
    assert any(r.value == "1" for r in requests)


def test_cluster_accelerators_unchanged():
    accelerator = get_proto_extended_resources(flyte.Resources(gpu="A100:1")).gpu_accelerator
    assert accelerator.device == "nvidia-tesla-a100"
    assert accelerator.device_class == 0  # NVIDIA_GPU


@pytest.mark.parametrize(
    "code, cls",
    [
        ("APPLE_SILICON_NOT_AVAILABLE", flyte.errors.AppleSiliconNotAvailableError),
        ("APPLE_SILICON_HOST_EXEC_FAILED", flyte.errors.AppleSiliconHostExecFailedError),
        ("APPLE_SILICON_DAEMON_START_FAILED", flyte.errors.AppleSiliconDaemonStartFailedError),
        ("APPLE_SILICON_DAEMON_OOM", flyte.errors.AppleSiliconDaemonOOMError),
        ("APPLE_SILICON_MODEL_NOT_FOUND", flyte.errors.AppleSiliconModelNotFoundError),
        ("APPLE_SILICON_BRIDGE_UNREACHABLE", flyte.errors.AppleSiliconBridgeUnreachableError),
    ],
)
def test_backend_codes_become_named_errors(code, cls):
    err = execution_pb2.ExecutionError(kind=execution_pb2.ExecutionError.USER, code=code, message="why")
    native = convert_error_to_native(err)
    assert type(native) is cls
    assert native.code == code
    assert isinstance(native, flyte.errors.RuntimeUserError)


@pytest.mark.parametrize(
    "system, machine, arm64_hw, expected",
    [
        ("Darwin", "arm64", True, None),
        ("Darwin", "x86_64", True, None),  # x86_64 Python under Rosetta, Apple Silicon hardware
        ("Darwin", "x86_64", False, "this host is darwin/x86_64 (Intel Mac)"),
        ("Linux", "x86_64", False, "this host is linux/x86_64"),
        ("Windows", "AMD64", False, "this host is windows/amd64"),
    ],
)
def test_apple_silicon_unavailable_reason(monkeypatch, system, machine, arm64_hw, expected):
    monkeypatch.setattr(_host.platform, "system", lambda: system)
    monkeypatch.setattr(_host.platform, "machine", lambda: machine)
    monkeypatch.setattr(_host, "_is_arm64_hardware", lambda: machine == "arm64" or arm64_hw)
    reason = _host.apple_silicon_unavailable_reason()
    if expected is None:
        assert reason is None
    else:
        assert reason.startswith("gpu='AppleSilicon' requires an Apple Silicon devbox") and expected in reason


def test_devbox_container_env(monkeypatch, tmp_path):
    monkeypatch.setattr(_host, "STATE_DIR", tmp_path)
    monkeypatch.setattr(_host, "host_platform", lambda: "darwin/arm64")
    monkeypatch.setattr(_host, "load_or_create_secret", lambda state_dir=None: "s3cret")
    assert _host.devbox_container_env(None) == {"FLYTE_DEVBOX_HOST_PLATFORM": "darwin/arm64"}
    assert _host.devbox_container_env(30090) == {
        "FLYTE_DEVBOX_HOST_PLATFORM": "darwin/arm64",
        "FLYTE_DEVBOX_APPLE_SILICON_PORT": "30090",
        "FLYTE_DEVBOX_APPLE_SILICON_TOKEN_SECRET": "s3cret",
    }


def test_secret_is_stable_and_private(tmp_path):
    first = _host.load_or_create_secret(tmp_path)
    assert _host.load_or_create_secret(tmp_path) == first
    assert (tmp_path / "secret").stat().st_mode & 0o777 == 0o600


def _host_can(monkeypatch, apple: bool, uv: bool = True):
    import shutil

    import flyte.cli._devbox as devbox

    monkeypatch.setattr(_host, "is_apple_silicon", lambda: apple)
    monkeypatch.setattr(
        _host,
        "apple_silicon_unavailable_reason",
        lambda: None if apple else "gpu='AppleSilicon' requires an Apple Silicon devbox",
    )
    real_which = shutil.which
    monkeypatch.setattr(
        devbox.shutil, "which", lambda name: ("/usr/bin/uv" if uv else None) if name == "uv" else real_which(name)
    )


def test_apple_silicon_is_opt_in(monkeypatch):
    from flyte.cli._devbox import _resolve_apple_silicon

    _host_can(monkeypatch, apple=True)
    assert _resolve_apple_silicon(False, is_dev_mode=False) is None
    assert _resolve_apple_silicon(True, is_dev_mode=False) == _host.DEFAULT_BRIDGE_PORT


@pytest.mark.parametrize(
    "apple, uv, dev, reason",
    [
        (False, True, False, "requires an Apple Silicon devbox"),
        (True, False, False, "uv"),  # host execution builds every environment with it; mlx-lm is optional
        (True, True, True, "dev mode"),
    ],
)
def test_apple_silicon_where_it_cannot_work_is_an_error(monkeypatch, apple, uv, dev, reason):
    import click

    from flyte.cli._devbox import _resolve_apple_silicon

    _host_can(monkeypatch, apple=apple, uv=uv)
    with pytest.raises(click.ClickException, match=reason):
        _resolve_apple_silicon(True, is_dev_mode=dev)


def test_start_devbox_flag(monkeypatch):
    from click.testing import CliRunner

    import flyte.cli._devbox as devbox
    from flyte.cli.main import main

    calls = []
    monkeypatch.setattr(devbox, "launch_devbox", lambda *a, **kw: calls.append(kw))
    assert CliRunner().invoke(main, ["start", "devbox", "--apple-silicon"]).exit_code == 0
    assert CliRunner().invoke(main, ["start", "devbox"]).exit_code == 0
    assert [c["apple_silicon"] for c in calls] == [True, False]
    result = CliRunner().invoke(main, ["start", "devbox", "--allow-host-execution"])
    assert result.exit_code != 0 and "No such option" in result.output


def _paused_container(host_ip: str, apple_silicon_port: str | None = "30090") -> dict:
    # A paused container keeps its configured bindings but reports no live NetworkSettings.Ports.
    env = [f"FLYTE_DEVBOX_APPLE_SILICON_PORT={apple_silicon_port}"] if apple_silicon_port else []
    return {
        "Config": {"Env": env},
        "HostConfig": {"PortBindings": {"30080/tcp": [{"HostIp": host_ip, "HostPort": "30080"}]}},
        "NetworkSettings": {"Ports": {}},
    }


def test_resume_restarts_the_bridge_the_devbox_was_created_with(monkeypatch):
    import flyte.cli._devbox as devbox

    started = []
    monkeypatch.setattr(devbox, "_start_apple_silicon_bridge", started.append)
    monkeypatch.setattr(devbox, "_inspect_container", lambda name: _paused_container("127.0.0.1"))
    devbox._restart_apple_silicon_bridge(None)
    assert started == [30090]

    monkeypatch.setattr(devbox, "_inspect_container", lambda name: _paused_container("127.0.0.1", None))
    devbox._restart_apple_silicon_bridge(None)
    assert started == [30090]  # a devbox created without --apple-silicon gets no bridge


def test_resume_refuses_a_devbox_reachable_from_the_network(monkeypatch):
    import click

    import flyte.cli._devbox as devbox

    monkeypatch.setattr(devbox, "_start_apple_silicon_bridge", lambda port: None)
    monkeypatch.setattr(devbox, "_inspect_container", lambda name: _paused_container("0.0.0.0"))
    with pytest.raises(click.ClickException, match="reachable from this machine only"):
        devbox._restart_apple_silicon_bridge(30090)
