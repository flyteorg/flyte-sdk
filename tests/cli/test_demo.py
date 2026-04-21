"""
Unit tests for flyte.cli._demo.

Covers the `--gpu` plumbing on `flyte start demo` and the
kubeconfig chown-retry fallback when kubectl fails to read a root-owned
kubeconfig on Linux bind mounts.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from flyte.cli._demo import _merge_kubeconfig, _run_container
from flyte.cli._start import demo


class TestRunContainerGpuFlag:
    """Verify the --gpu flag appends `--gpus all` to the docker run command."""

    @staticmethod
    def _invoke(gpu: bool) -> list[str]:
        with patch("flyte.cli._demo.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run_container(
                image="ghcr.io/flyteorg/flyte-demo:gpu-latest",
                is_dev_mode=False,
                container_name="flyte-demo",
                kube_dir=Path("/tmp/.kube"),
                flyte_demo_config_dir=Path("/tmp/.flyte/demo"),
                volume_name="flyte-demo",
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
        # `docker run [options] <image>` — --gpus must come before the image arg.
        cmd = self._invoke(gpu=True)
        assert cmd.index("--gpus") < cmd.index("ghcr.io/flyteorg/flyte-demo:gpu-latest")


class TestMergeKubeconfigRetry:
    """Verify the chown-retry fallback for a root-owned kubeconfig on Linux."""

    def test_success_on_first_try_does_not_chown(self, tmp_path):
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("")

        with (
            patch("flyte.cli._demo._flatten_kubeconfig") as mock_flatten,
            patch("flyte.cli._demo.subprocess.run") as mock_run,
            patch("flyte.cli._demo.shutil.move", side_effect=lambda src, dst: Path(dst).touch()),
            patch("flyte.cli._demo.Path.home", return_value=tmp_path),
        ):
            mock_flatten.return_value = MagicMock(stdout="apiVersion: v1\n")

            _merge_kubeconfig(kubeconfig, "flyte-demo")

            assert mock_flatten.call_count == 1
            mock_run.assert_not_called()

    def test_called_process_error_triggers_chown_and_retry(self, tmp_path):
        """This is the bug fix: on Linux, kubectl exits non-zero (CalledProcessError),
        not PermissionError. The retry branch must fire."""
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("")

        with (
            patch("flyte.cli._demo._flatten_kubeconfig") as mock_flatten,
            patch("flyte.cli._demo.subprocess.run") as mock_run,
            patch("flyte.cli._demo.shutil.move", side_effect=lambda src, dst: Path(dst).touch()),
            patch("flyte.cli._demo.Path.home", return_value=tmp_path),
        ):
            mock_flatten.side_effect = [
                subprocess.CalledProcessError(1, ["kubectl", "config", "view", "--flatten"]),
                MagicMock(stdout="apiVersion: v1\n"),
            ]

            _merge_kubeconfig(kubeconfig, "flyte-demo")

            assert mock_flatten.call_count == 2
            assert mock_run.call_count == 1
            docker_cmd = mock_run.call_args.args[0]
            assert docker_cmd[:4] == ["docker", "exec", "flyte-demo", "chown"]
            assert docker_cmd[-1] == "/.kube/kubeconfig"

    def test_permission_error_still_triggers_chown_and_retry(self, tmp_path):
        """Legacy path — macOS users opening the file directly — should still work."""
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("")

        with (
            patch("flyte.cli._demo._flatten_kubeconfig") as mock_flatten,
            patch("flyte.cli._demo.subprocess.run") as mock_run,
            patch("flyte.cli._demo.shutil.move", side_effect=lambda src, dst: Path(dst).touch()),
            patch("flyte.cli._demo.Path.home", return_value=tmp_path),
        ):
            mock_flatten.side_effect = [
                PermissionError("denied"),
                MagicMock(stdout="apiVersion: v1\n"),
            ]

            _merge_kubeconfig(kubeconfig, "flyte-demo")

            assert mock_flatten.call_count == 2
            assert mock_run.call_count == 1

    def test_second_flatten_failure_propagates(self, tmp_path):
        """If kubectl still fails after the chown, we should not swallow the error."""
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("")

        with (
            patch("flyte.cli._demo._flatten_kubeconfig") as mock_flatten,
            patch("flyte.cli._demo.subprocess.run"),
            patch("flyte.cli._demo.Path.home", return_value=tmp_path),
        ):
            err = subprocess.CalledProcessError(1, ["kubectl"])
            mock_flatten.side_effect = [err, err]

            with pytest.raises(subprocess.CalledProcessError):
                _merge_kubeconfig(kubeconfig, "flyte-demo")


class TestDemoCliGpuFlag:
    """Verify the --gpu Click option is plumbed to launch_demo."""

    def test_gpu_flag_passed_through(self):
        runner = CliRunner()
        with patch("flyte.cli._demo.launch_demo") as mock_launch:
            result = runner.invoke(demo, ["--gpu", "--image", "flyte-demo:gpu-latest"])
            assert result.exit_code == 0, result.output
            mock_launch.assert_called_once()
            assert mock_launch.call_args.kwargs["gpu"] is True

    def test_gpu_defaults_to_false(self):
        runner = CliRunner()
        with patch("flyte.cli._demo.launch_demo") as mock_launch:
            result = runner.invoke(demo, ["--image", "flyte-demo:latest"])
            assert result.exit_code == 0, result.output
            mock_launch.assert_called_once()
            assert mock_launch.call_args.kwargs["gpu"] is False
