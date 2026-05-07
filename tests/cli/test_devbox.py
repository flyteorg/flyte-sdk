"""
Unit tests for flyte.cli._devbox.

Covers the `--gpu` plumbing on `flyte start devbox` and the
kubeconfig chown-retry fallback when kubectl fails to read a root-owned
kubeconfig on Linux bind mounts.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from flyte.cli._devbox import _merge_kubeconfig, _run_container
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
