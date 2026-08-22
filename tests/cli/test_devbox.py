"""
Unit tests for flyte.cli._devbox.

Covers the `--gpu` plumbing on `flyte start devbox`, the kubeconfig chown-retry
fallback when kubectl fails to read a root-owned kubeconfig on Linux bind mounts,
and the status snapshot behind `flyte get devbox`.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from flyte.cli._devbox import (
    _is_kubectl_installed,
    _merge_kubeconfig,
    _run_container,
    _switch_k8s_context,
)
from flyte.cli._start import devbox


class TestRunContainerGpuFlag:
    """Verify the --gpu flag appends `--gpus all` to the docker run command."""

    @staticmethod
    def _invoke(gpu: bool) -> list[str]:
        with patch("flyte.cli._devbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run_container(
                image="ghcr.io/flyteorg/flyte-devbox:gpu-latest",
                is_dev_mode=False,
                container_name="flyte-devbox",
                kube_dir=Path("/tmp/.kube"),
                flyte_devbox_config_dir=Path("/tmp/.flyte/devbox"),
                volume_name="flyte-devbox",
                ports=["30080:30080"],
                gpu=gpu,
            )
            assert mock_run.call_count == 1
            return mock_run.call_args.args[0]

    def test_gpu_flag_appends_gpus_all(self):
        cmd = self._invoke(gpu=True)
        assert "--gpus" in cmd
        assert cmd[cmd.index("--gpus") + 1] == "all"

    def test_gpu_disabled_does_not_set_gpus(self):
        cmd = self._invoke(gpu=False)
        assert "--gpus" not in cmd

    def test_gpu_flag_precedes_image(self):
        cmd = self._invoke(gpu=True)
        assert cmd.index("--gpus") < cmd.index("ghcr.io/flyteorg/flyte-devbox:gpu-latest")

    def test_kube_dir_mount_resolves_symlinks(self, tmp_path):
        kube_dir = tmp_path / "real-kube-dir"
        kube_dir.mkdir()
        kube_dir_symlink = tmp_path / "kube-dir-symlink"
        kube_dir_symlink.symlink_to(kube_dir, target_is_directory=True)

        with patch("flyte.cli._devbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run_container(
                image="flyte-devbox:latest",
                is_dev_mode=False,
                container_name="flyte-devbox",
                kube_dir=kube_dir_symlink,
                flyte_devbox_config_dir=tmp_path / "devbox",
                volume_name="flyte-devbox",
                ports=[],
            )

        cmd = mock_run.call_args.args[0]
        assert f"{kube_dir}:/.kube" in cmd


class TestMergeKubeconfigRetry:
    """Verify the chown-retry fallback for a root-owned kubeconfig on Linux."""

    def test_success_on_first_try_does_not_chown(self, tmp_path):
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("")

        with (
            patch("flyte.cli._devbox._flatten_kubeconfig") as mock_flatten,
            patch("flyte.cli._devbox.subprocess.run") as mock_run,
            patch("flyte.cli._devbox.shutil.move", side_effect=lambda src, dst: Path(dst).touch()),
            patch("flyte.cli._devbox.Path.home", return_value=tmp_path),
        ):
            mock_flatten.return_value = MagicMock(stdout="apiVersion: v1\n")

            _merge_kubeconfig(kubeconfig, "flyte-devbox")

            assert mock_flatten.call_count == 1
            mock_run.assert_not_called()

    def test_called_process_error_triggers_chown_and_retry(self, tmp_path):
        """This is the bug fix: on Linux, kubectl exits non-zero (CalledProcessError),
        not PermissionError. The retry branch must fire."""
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("")

        with (
            patch("flyte.cli._devbox._flatten_kubeconfig") as mock_flatten,
            patch("flyte.cli._devbox.subprocess.run") as mock_run,
            patch("flyte.cli._devbox.shutil.move", side_effect=lambda src, dst: Path(dst).touch()),
            patch("flyte.cli._devbox.Path.home", return_value=tmp_path),
        ):
            mock_flatten.side_effect = [
                subprocess.CalledProcessError(1, ["kubectl", "config", "view", "--flatten"]),
                MagicMock(stdout="apiVersion: v1\n"),
            ]

            _merge_kubeconfig(kubeconfig, "flyte-devbox")

            assert mock_flatten.call_count == 2
            assert mock_run.call_count == 1
            docker_cmd = mock_run.call_args.args[0]
            assert docker_cmd[:4] == ["docker", "exec", "flyte-devbox", "chown"]
            assert docker_cmd[-1] == "/.kube/kubeconfig"

    def test_permission_error_still_triggers_chown_and_retry(self, tmp_path):
        """Legacy path — macOS users opening the file directly — should still work."""
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("")

        with (
            patch("flyte.cli._devbox._flatten_kubeconfig") as mock_flatten,
            patch("flyte.cli._devbox.subprocess.run") as mock_run,
            patch("flyte.cli._devbox.shutil.move", side_effect=lambda src, dst: Path(dst).touch()),
            patch("flyte.cli._devbox.Path.home", return_value=tmp_path),
        ):
            mock_flatten.side_effect = [
                PermissionError("denied"),
                MagicMock(stdout="apiVersion: v1\n"),
            ]

            _merge_kubeconfig(kubeconfig, "flyte-devbox")

            assert mock_flatten.call_count == 2
            assert mock_run.call_count == 1

    def test_second_flatten_failure_propagates(self, tmp_path):
        """If kubectl still fails after the chown, we should not swallow the error."""
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("")

        with (
            patch("flyte.cli._devbox._flatten_kubeconfig") as mock_flatten,
            patch("flyte.cli._devbox.subprocess.run"),
            patch("flyte.cli._devbox.Path.home", return_value=tmp_path),
        ):
            err = subprocess.CalledProcessError(1, ["kubectl"])
            mock_flatten.side_effect = [err, err]

            with pytest.raises(subprocess.CalledProcessError):
                _merge_kubeconfig(kubeconfig, "flyte-devbox")


class TestIsKubectlInstalled:
    """`_is_kubectl_installed` reports kubectl presence as a bool instead of raising,
    and callers skip kubectl-dependent steps (with a warning) when it is missing."""

    def test_missing_kubectl_returns_false(self):
        with patch("flyte.cli._devbox.shutil.which", return_value=None):
            assert _is_kubectl_installed() is False

    def test_present_kubectl_returns_true(self):
        with patch("flyte.cli._devbox.shutil.which", return_value="/usr/local/bin/kubectl"):
            assert _is_kubectl_installed() is True

    def test_merge_kubeconfig_skips_and_warns_when_kubectl_missing(self, tmp_path):
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("")
        with (
            patch("flyte.cli._devbox._is_kubectl_installed", return_value=False),
            patch("flyte.cli._devbox._flatten_kubeconfig") as mock_flatten,
            patch("flyte.cli._devbox.console.print") as mock_print,
            patch("flyte.cli._devbox.Path.home", return_value=tmp_path),
        ):
            # Should not raise, and should not attempt to flatten/merge.
            _merge_kubeconfig(kubeconfig, "flyte-devbox")

            mock_flatten.assert_not_called()
            mock_print.assert_called_once()
            message = mock_print.call_args.args[0]
            assert "kubectl" in message
            assert "[red]" in message

    def test_switch_k8s_context_skips_and_warns_when_kubectl_missing(self):
        with (
            patch("flyte.cli._devbox._is_kubectl_installed", return_value=False),
            patch("flyte.cli._devbox.subprocess.run") as mock_run,
            patch("flyte.cli._devbox.console.print") as mock_print,
        ):
            # Should not raise, and should never shell out to kubectl.
            _switch_k8s_context()

            mock_run.assert_not_called()
            mock_print.assert_called_once()
            message = mock_print.call_args.args[0]
            assert "kubectl" in message
            assert "[red]" in message


class TestDevboxCliGpuFlag:
    """Verify the --gpu Click option is plumbed to launch_devbox."""

    def test_gpu_flag_passed_through(self):
        runner = CliRunner()
        with patch("flyte.cli._devbox.launch_devbox") as mock_launch:
            result = runner.invoke(devbox, ["--gpu", "--image", "flyte-devbox:gpu-latest"])
            assert result.exit_code == 0, result.output
            mock_launch.assert_called_once()
            assert mock_launch.call_args.kwargs["gpu"] is True

    def test_gpu_defaults_to_false(self):
        runner = CliRunner()
        with patch("flyte.cli._devbox.launch_devbox") as mock_launch:
            result = runner.invoke(devbox, ["--image", "flyte-devbox:latest"])
            assert result.exit_code == 0, result.output
            mock_launch.assert_called_once()
            assert mock_launch.call_args.kwargs["gpu"] is False


class TestDevboxCliDefaultImage:
    """--gpu without --image should pick the GPU-capable default image."""

    def test_gpu_without_image_uses_gpu_default(self):
        from flyte.cli._start import _DEFAULT_DEVBOX_GPU_IMAGE

        runner = CliRunner()
        with patch("flyte.cli._devbox.launch_devbox") as mock_launch:
            result = runner.invoke(devbox, ["--gpu"])
            assert result.exit_code == 0, result.output
            assert mock_launch.call_args.args[0] == _DEFAULT_DEVBOX_GPU_IMAGE

    def test_no_flags_uses_cpu_default(self):
        from flyte.cli._start import _DEFAULT_DEVBOX_IMAGE

        runner = CliRunner()
        with patch("flyte.cli._devbox.launch_devbox") as mock_launch:
            result = runner.invoke(devbox, [])
            assert result.exit_code == 0, result.output
            assert mock_launch.call_args.args[0] == _DEFAULT_DEVBOX_IMAGE

    def test_explicit_image_with_gpu_is_respected(self):
        runner = CliRunner()
        with patch("flyte.cli._devbox.launch_devbox") as mock_launch:
            result = runner.invoke(devbox, ["--gpu", "--image", "myorg/custom:latest"])
            assert result.exit_code == 0, result.output
            assert mock_launch.call_args.args[0] == "myorg/custom:latest"


class TestDockerSubprocessFailures:
    """Docker CLI failures should surface as click.ClickException, not raw CalledProcessError."""

    def test_ensure_volume_failure_raises_click_exception(self):
        import click

        from flyte.cli._devbox import _ensure_volume

        with patch("flyte.cli._devbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="docker daemon not reachable")
            with pytest.raises(click.ClickException) as excinfo:
                _ensure_volume("flyte-devbox")
            assert "Failed to list docker volumes" in str(excinfo.value.message)
            assert "docker daemon not reachable" in str(excinfo.value.message)

    def test_container_is_running_failure_raises_click_exception(self):
        import click

        from flyte.cli._devbox import _container_is_running

        with patch("flyte.cli._devbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
            with pytest.raises(click.ClickException):
                _container_is_running("flyte-devbox")

    def test_container_is_paused_failure_raises_click_exception(self):
        import click

        from flyte.cli._devbox import _container_is_paused

        with patch("flyte.cli._devbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
            with pytest.raises(click.ClickException):
                _container_is_paused("flyte-devbox")


class TestHealthUrl:
    """The devbox readiness probe must hit /healthz, not /readyz."""

    def test_health_url_uses_healthz(self):
        from flyte.cli._devbox import _health_url

        assert _health_url() == "http://localhost:30080/healthz"
        assert _health_url(41080) == "http://localhost:41080/healthz"

    def test_launch_waits_on_healthz(self):
        """`flyte start devbox` polls the same endpoint; /readyz is shadowed by the object
        store ingress and answers 403 on a healthy cluster, which hangs the wait."""
        from flyte.cli._devbox import _wait_for_devbox_ready

        with patch("flyte.cli._devbox._wait_for_console_ready") as mock_wait:
            _wait_for_devbox_ready(is_dev_mode=False)

        mock_wait.assert_called_once_with("http://localhost:30080/healthz")

    def test_status_probe_uses_healthz(self):
        from flyte.cli._devbox import get_devbox_status

        with (
            patch("flyte.cli._devbox._docker_unavailable_reason", return_value=None),
            patch("flyte.cli._devbox._inspect_container", return_value=_RUNNING_INSPECT),
            patch("flyte.cli._devbox._inspect_image", return_value=None),
            patch("flyte.cli._devbox._inspect_volume", return_value=None),
            patch("flyte.cli._devbox._container_stats", return_value=None),
            patch("flyte.cli._devbox._kube_context_state", return_value=(None, None)),
            patch("flyte.cli._devbox._console_is_ready", return_value=True) as mock_ready,
        ):
            status = get_devbox_status()

        assert status.ready is True
        assert mock_ready.call_args.args[0] == "http://localhost:30080/healthz"


class TestParseDockerTime:
    """Docker emits RFC3339 with nanosecond precision, which `datetime` cannot parse directly."""

    def test_nanosecond_precision_is_truncated(self):
        from flyte.cli._devbox import _parse_docker_time

        parsed = _parse_docker_time("2026-08-17T22:29:20.426143387Z")
        assert parsed is not None
        assert parsed.year == 2026
        assert parsed.microsecond == 426143
        assert parsed.utcoffset().total_seconds() == 0

    def test_offset_timezone_is_preserved(self):
        from flyte.cli._devbox import _parse_docker_time

        parsed = _parse_docker_time("2026-08-17T18:29:20.426143387-04:00")
        assert parsed is not None
        assert parsed.utcoffset().total_seconds() == -4 * 3600

    def test_no_fractional_seconds(self):
        from flyte.cli._devbox import _parse_docker_time

        assert _parse_docker_time("2026-08-17T22:29:20Z") is not None

    def test_zero_value_and_garbage_return_none(self):
        from flyte.cli._devbox import _parse_docker_time

        assert _parse_docker_time("0001-01-01T00:00:00Z") is None
        assert _parse_docker_time(None) is None
        assert _parse_docker_time("not-a-time") is None


class TestFormatHelpers:
    def test_humanize_duration(self):
        from flyte.cli._devbox import _humanize_duration

        assert _humanize_duration(42) == "42s"
        assert _humanize_duration(125) == "2m 5s"
        assert _humanize_duration(3 * 3600 + 120) == "3h 2m"
        assert _humanize_duration(50 * 3600) == "2d 2h"

    def test_humanize_size(self):
        from flyte.cli._devbox import _humanize_size

        assert _humanize_size(None) is None
        assert _humanize_size(512) == "512 B"
        assert _humanize_size(532014052) == "507.4 MiB"

    def test_port_bindings_skips_unpublished_ports(self):
        from flyte.cli._devbox import _port_bindings

        inspect = {
            "NetworkSettings": {
                "Ports": {
                    "30080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "31080"}],
                    "6443/tcp": None,
                }
            }
        }
        assert _port_bindings(inspect) == {"30080/tcp": "31080"}

    def test_port_bindings_missing_section(self):
        from flyte.cli._devbox import _port_bindings

        assert _port_bindings({}) == {}

    def test_container_env(self):
        from flyte.cli._devbox import _container_env

        env = _container_env({"Config": {"Env": ["FLYTE_DEV=True", "PATH=/usr/bin"]}})
        assert env["FLYTE_DEV"] == "True"
        assert env["PATH"] == "/usr/bin"

    def test_port_number_sorts_numerically(self):
        from flyte.cli._devbox import _port_number

        assert sorted(["30080/tcp", "6443/tcp", "30000/tcp"], key=_port_number) == [
            "6443/tcp",
            "30000/tcp",
            "30080/tcp",
        ]

    def test_image_label_summary(self):
        from flyte.cli._devbox import _image_label_summary

        assert _image_label_summary({}) is None
        assert (
            _image_label_summary(
                {
                    "org.opencontainers.image.title": "k3s",
                    "org.opencontainers.image.version": "v1.34.6-k3s1",
                    "org.opencontainers.image.revision": "234e61326ca4e005522be1e69645c1ca5754121f",
                }
            )
            == "k3s v1.34.6-k3s1 (rev 234e61326ca4)"
        )
        assert _image_label_summary({"org.opencontainers.image.version": "1.2.3"}) == "1.2.3"


_RUNNING_INSPECT = {
    "Id": "89d7d1d8ec0a62fd3c5aa0b307652192e947f1128445f272abd736c11dfcd9e6",
    "Created": "2026-08-17T22:29:18.566280678Z",
    "State": {"Status": "running", "StartedAt": "2026-08-17T22:29:20.426143387Z"},
    "Config": {"Image": "cr.flyte.org/flyteorg/flyte-devbox:latest", "Env": ["FLYTE_DEV=True"]},
    "HostConfig": {"DeviceRequests": [{"Driver": "", "Count": -1, "Capabilities": [["gpu"]]}]},
    "NetworkSettings": {
        "Ports": {
            "30080/tcp": [{"HostPort": "30080"}],
            "30000/tcp": [{"HostPort": "30000"}],
            "6443/tcp": [{"HostPort": "6443"}],
        }
    },
}

_IMAGE_INSPECT = {
    "Id": "sha256:95fd805f48388d8d4e609a0f86109f3323c22f35742339834e06e65bada2c07e",
    "RepoDigests": ["cr.flyte.org/flyteorg/flyte-devbox@sha256:95fd805f4838"],
    "Created": "2026-08-15T00:31:02.521722319Z",
    "Size": 532014052,
    "Config": {"Labels": {"org.opencontainers.image.version": "v1.34.6-k3s1"}},
}


class TestGetDevboxStatus:
    """`get_devbox_status` degrades gracefully: every probe is optional."""

    @staticmethod
    def _patched(container, volume=None, ready=True, stats=None, kube=("flyte-devbox", True)):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(patch("flyte.cli._devbox.shutil.which", return_value="/usr/local/bin/docker"))
        stack.enter_context(patch("flyte.cli._devbox.subprocess.run", return_value=MagicMock(returncode=0)))
        stack.enter_context(patch("flyte.cli._devbox._inspect_container", return_value=container))
        stack.enter_context(patch("flyte.cli._devbox._inspect_image", return_value=_IMAGE_INSPECT))
        stack.enter_context(patch("flyte.cli._devbox._inspect_volume", return_value=volume))
        stack.enter_context(patch("flyte.cli._devbox._console_is_ready", return_value=ready))
        stack.enter_context(patch("flyte.cli._devbox._container_stats", return_value=stats))
        stack.enter_context(patch("flyte.cli._devbox._kube_context_state", return_value=kube))
        return stack

    def test_docker_missing(self):
        from flyte.cli._devbox import get_devbox_status

        with patch("flyte.cli._devbox.shutil.which", return_value=None):
            status = get_devbox_status()

        assert status.state == "docker-unavailable"
        assert "Docker is not installed" in status.error
        assert status.is_running is False

    def test_docker_daemon_not_running(self):
        from flyte.cli._devbox import get_devbox_status

        with (
            patch("flyte.cli._devbox.shutil.which", return_value="/usr/local/bin/docker"),
            patch(
                "flyte.cli._devbox.subprocess.run",
                return_value=MagicMock(returncode=1, stdout="", stderr="Cannot connect to the Docker daemon"),
            ),
        ):
            status = get_devbox_status()

        assert status.state == "docker-unavailable"
        assert "Cannot connect to the Docker daemon" in status.error

    def test_container_not_found_still_reports_volume(self):
        from flyte.cli._devbox import get_devbox_status

        with self._patched(container=None, volume={"Mountpoint": "/var/lib/docker/volumes/flyte-devbox/_data"}):
            status = get_devbox_status()

        assert status.state == "not-found"
        assert status.is_running is False
        assert status.volume_exists is True
        assert status.volume_mountpoint == "/var/lib/docker/volumes/flyte-devbox/_data"
        assert status.kube_context == "flyte-devbox"
        assert status.container_id is None
        assert status.ready is None
        assert status.endpoint is None

    def test_stopped_container_reports_no_endpoints(self):
        import copy

        from flyte.cli._devbox import get_devbox_status

        inspect = copy.deepcopy(_RUNNING_INSPECT)
        inspect["State"] = {"Status": "exited", "StartedAt": "2026-08-17T22:29:20.426143387Z"}
        inspect["NetworkSettings"] = {"Ports": {}}

        with self._patched(container=inspect):
            status = get_devbox_status()

        assert status.state == "exited"
        assert status.endpoint is None
        assert status.console_url is None
        assert status.registry is None
        assert status.uptime_seconds is None
        assert status.image == "cr.flyte.org/flyteorg/flyte-devbox:latest"

    def test_running_container_is_fully_described(self):
        from flyte.cli._devbox import get_devbox_status

        with self._patched(
            container=_RUNNING_INSPECT,
            volume={"Mountpoint": "/data"},
            stats={"CPUPerc": "1.50%", "MemUsage": "1.2GiB / 31.3GiB"},
        ):
            status = get_devbox_status()

        assert status.state == "running"
        assert status.is_running is True
        assert status.container_id == "89d7d1d8ec0a"
        assert status.dev_mode is True
        assert status.gpu is True
        assert status.image == "cr.flyte.org/flyteorg/flyte-devbox:latest"
        assert status.image_id == "95fd805f4838"
        assert status.image_digest == "cr.flyte.org/flyteorg/flyte-devbox@sha256:95fd805f4838"
        assert status.image_size_bytes == 532014052
        assert status.console_url == "http://localhost:30080/v2"
        assert status.endpoint == "localhost:30080"
        assert status.registry == "localhost:30000"
        assert status.k8s_api_url == "https://localhost:6443"
        assert status.ready is True
        assert status.cpu_percent == "1.50%"
        assert status.memory_usage == "1.2GiB / 31.3GiB"
        assert status.kube_context == "flyte-devbox"
        assert status.kube_context_is_current is True
        assert status.uptime_seconds is not None and status.uptime_seconds > 0

    def test_endpoints_follow_remapped_host_ports(self):
        """A container published on non-default host ports should report those ports."""
        import copy

        from flyte.cli._devbox import get_devbox_status

        inspect = copy.deepcopy(_RUNNING_INSPECT)
        inspect["NetworkSettings"]["Ports"]["30080/tcp"] = [{"HostPort": "41080"}]
        inspect["NetworkSettings"]["Ports"]["30000/tcp"] = [{"HostPort": "41000"}]

        with self._patched(container=inspect):
            status = get_devbox_status()

        assert status.console_url == "http://localhost:41080/v2"
        assert status.endpoint == "localhost:41080"
        assert status.registry == "localhost:41000"

    def test_ready_and_stats_checks_can_be_skipped(self):
        from flyte.cli._devbox import get_devbox_status

        with self._patched(container=_RUNNING_INSPECT, stats={"CPUPerc": "1.50%"}):
            status = get_devbox_status(check_ready=False, check_stats=False)

        assert status.ready is None
        assert status.cpu_percent is None

    def test_paused_container_reports_no_readiness_probe(self):
        import copy

        from flyte.cli._devbox import get_devbox_status

        inspect = copy.deepcopy(_RUNNING_INSPECT)
        inspect["State"] = {"Status": "paused", "StartedAt": "2026-08-17T22:29:20.426143387Z"}

        with self._patched(container=inspect):
            status = get_devbox_status()

        assert status.state == "paused"
        assert status.is_running is False
        assert status.ready is None
        assert status.uptime_seconds is not None


class TestRenderDevboxStatus:
    @staticmethod
    def _render(status) -> str:
        from rich.console import Console

        from flyte.cli._devbox import render_devbox_status

        console = Console(force_terminal=False, width=120, no_color=True)
        with console.capture() as capture:
            console.print(render_devbox_status(status))
        return capture.get()

    def test_running_panel_shows_endpoints_and_version(self):
        from flyte.cli._devbox import DevboxStatus

        text = self._render(
            DevboxStatus(
                state="running",
                container_id="89d7d1d8ec0a",
                image="cr.flyte.org/flyteorg/flyte-devbox:latest",
                image_id="95fd805f4838",
                console_url="http://localhost:30080/v2",
                endpoint="localhost:30080",
                registry="localhost:30000",
                ready=True,
                uptime_seconds=125,
            )
        )
        assert "Running" in text
        assert "up 2m 5s" in text
        assert "localhost:30080" in text
        assert "localhost:30000" in text
        assert "95fd805f4838" in text

    def test_not_found_panel_suggests_start(self):
        from flyte.cli._devbox import DevboxStatus

        text = self._render(DevboxStatus(state="not-found"))
        assert "Not running" in text
        assert "flyte start devbox" in text

    def test_paused_panel_suggests_resume(self):
        from flyte.cli._devbox import DevboxStatus

        text = self._render(DevboxStatus(state="paused", uptime_seconds=125))
        assert "Paused" in text
        assert "started 2m 5s ago" in text
        assert "flyte start devbox" in text

    def test_docker_unavailable_panel_shows_error(self):
        from flyte.cli._devbox import DevboxStatus

        text = self._render(DevboxStatus(state="docker-unavailable", error="Docker daemon is not reachable."))
        assert "Docker unavailable" in text
        assert "Docker daemon is not reachable." in text

    def test_unknown_state_is_passed_through(self):
        from flyte.cli._devbox import DevboxStatus

        text = self._render(DevboxStatus(state="dead"))
        assert "Dead" in text


class TestGetDevboxCommand:
    def test_table_output(self):
        from flyte.cli._devbox import DevboxStatus
        from flyte.cli._get import devbox as get_devbox

        runner = CliRunner()
        with patch(
            "flyte.cli._devbox.get_devbox_status",
            return_value=DevboxStatus(state="running", endpoint="localhost:30080", ready=True),
        ):
            result = runner.invoke(get_devbox, [])

        assert result.exit_code == 0, result.output
        assert "Running" in result.output

    @staticmethod
    def _invoke_with_format(output_format: str, status=None):
        from flyte.cli._common import CLIConfig
        from flyte.cli._devbox import DevboxStatus
        from flyte.cli._get import devbox as get_devbox

        cfg = MagicMock(spec=CLIConfig)
        cfg.output_format = output_format
        status = status or DevboxStatus(state="running", endpoint="localhost:30080", ready=True)
        with patch("flyte.cli._devbox.get_devbox_status", return_value=status):
            return CliRunner().invoke(get_devbox, [], obj=cfg)

    def test_json_raw_output_is_parseable(self):
        import json

        result = self._invoke_with_format("json-raw")

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload[0]["state"] == "running"
        assert payload[0]["endpoint"] == "localhost:30080"
        assert payload[0]["ready"] is True

    def test_json_output_renders_the_snapshot(self):
        import re

        result = self._invoke_with_format("json")

        assert result.exit_code == 0, result.output
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "'state': 'running'" in plain
        assert "'endpoint': 'localhost:30080'" in plain

    def test_no_probes_skips_readiness_and_stats(self):
        from flyte.cli._devbox import DevboxStatus
        from flyte.cli._get import devbox as get_devbox

        runner = CliRunner()
        with patch("flyte.cli._devbox.get_devbox_status", return_value=DevboxStatus(state="not-found")) as mock_status:
            result = runner.invoke(get_devbox, ["--no-probes"])

        assert result.exit_code == 0, result.output
        assert mock_status.call_args.kwargs == {"check_ready": False, "check_stats": False}


class TestPauseResume:
    """`docker pause`/`unpause` failures must not surface as raw CalledProcessError."""

    def test_resume_container_success(self):
        from flyte.cli._devbox import _resume_container

        with patch("flyte.cli._devbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="flyte-devbox\n", stderr="")
            assert _resume_container("flyte-devbox") is True
            assert mock_run.call_args.args[0] == ["docker", "unpause", "flyte-devbox"]

    def test_resume_container_not_paused_is_not_an_error(self):
        """The paused check and the unpause are separate docker calls; losing that race
        (`container not paused`) must fall through, not crash."""
        from flyte.cli._devbox import _resume_container

        with patch("flyte.cli._devbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr=(
                    "Error response from daemon: Cannot unpause container 06fcfce7dc99: "
                    "OCI runtime resume failed: container not paused\n"
                ),
            )
            assert _resume_container("flyte-devbox") is False

    def test_resume_container_other_failure_raises_click_exception(self):
        import click

        from flyte.cli._devbox import _resume_container

        with patch("flyte.cli._devbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="No such container: flyte-devbox")
            with pytest.raises(click.ClickException) as excinfo:
                _resume_container("flyte-devbox")
            assert "No such container" in str(excinfo.value.message)

    def test_launch_devbox_falls_through_when_resume_loses_the_race(self):
        """A container that stopped being paused should be treated like a fresh start."""
        from flyte.cli._devbox import launch_devbox

        with (
            patch("flyte.cli._devbox._ensure_docker_available"),
            patch("flyte.cli._devbox._ensure_volume"),
            patch("flyte.cli._devbox._container_is_paused", return_value=True),
            patch("flyte.cli._devbox._resume_container", return_value=False),
            patch("flyte.cli._devbox._container_is_running", return_value=False),
            patch("flyte.cli._devbox.subprocess.run"),
            patch("flyte.cli._devbox.Path.mkdir"),
            patch("flyte.cli._devbox.Path.unlink"),
            patch("flyte.cli._devbox._launch_devbox_rich") as mock_launch,
        ):
            launch_devbox("cr.flyte.org/flyteorg/flyte-devbox:latest", is_dev_mode=False)
            mock_launch.assert_called_once()

    def test_stop_devbox_pause_failure_raises_click_exception(self):
        import click

        from flyte.cli._devbox import stop_devbox

        with (
            patch("flyte.cli._devbox._container_is_paused", return_value=False),
            patch("flyte.cli._devbox._container_is_running", return_value=True),
            patch("flyte.cli._devbox.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="container is not running")
            with pytest.raises(click.ClickException) as excinfo:
                stop_devbox()
            assert "Failed to pause devbox container" in str(excinfo.value.message)
