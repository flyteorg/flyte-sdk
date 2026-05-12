import asyncio
import pathlib
import sys

import pytest

import flyte
from flyte.io import File
from flyte._pod import PodTemplate
from flyte.extras import ContainerTask


def test_bad_incorrect_type_in_command():
    run_name = "test_run"
    job_name = "test_job"
    # {"test_hyperparams": {"param1": 1, "param2": 2}}
    hyperparams_str = "eyJ0ZXN0X2h5cGFyYW1zIjogeyJwYXJhbTEiOiAxLCAicGFyYW0yIjogMn19"
    i = 10  # Incorrect type, should be str

    with pytest.raises(ValueError):
        ContainerTask(
            name="run_training",
            image="ghcr.io/dansola/test-image:python37-entrypoint",
            command=[
                "--run-name",
                run_name,
                "--job-name",
                job_name,
                "--file-suffix",
                i,
                "--hyperparams-base64",
                hyperparams_str,
            ],
        )


def test_block_network_default_is_false():
    task = ContainerTask(
        name="test",
        image="alpine:latest",
        command=["echo", "hi"],
    )
    assert task.pod_template is None
    assert task._block_network is False


def test_block_network_true_sets_pod_template():
    task = ContainerTask(
        name="test",
        image="alpine:latest",
        command=["echo", "hi"],
        block_network=True,
    )
    assert task.pod_template == "sandboxed-pod-template"
    assert task._block_network is True


def test_block_network_merges_label_into_pod_template():
    pt = PodTemplate(labels={"existing": "label"})
    task = ContainerTask(
        name="test",
        image="alpine:latest",
        command=["echo", "hi"],
        pod_template=pt,
        block_network=True,
    )
    assert task.pod_template.labels == {"existing": "label", "sandboxed": "true"}


def test_block_network_with_string_pod_template_raises():
    with pytest.raises(ValueError, match="block_network=True cannot be combined"):
        ContainerTask(
            name="test",
            image="alpine:latest",
            command=["echo", "hi"],
            pod_template="my-custom-template",
            block_network=True,
        )


def test_bad_incorrect_type_in_args():
    run_name = "test_run"
    job_name = "test_job"
    # {"test_hyperparams": {"param1": 1, "param2": 2}}
    hyperparams_str = "eyJ0ZXN0X2h5cGFyYW1zIjogeyJwYXJhbTEiOiAxLCAicGFyYW0yIjogMn19"
    i = 10  # Incorrect type, should be str
    f = 0.1  # Incorrect type, should be str

    with pytest.raises(ValueError):
        ContainerTask(
            name="run_training",
            image="ghcr.io/dansola/test-image:python37-entrypoint",
            command=["python", "train.py"],
            arguments=[
                "--run-name",
                run_name,
                "--job-name",
                job_name,
                "--file-suffix",
                i,
                "--dropout",
                f,
                "--hyperparams-base64",
                hyperparams_str,
            ],
        )


def test_local_execute_materializes_list_of_files(monkeypatch, tmp_path):
    flyte.init()
    src_a = tmp_path / "a.txt"
    src_b = tmp_path / "b.txt"
    src_a.write_text("alpha\n")
    src_b.write_text("beta\n")
    parts = [File.from_local_sync(str(src_a)), File.from_local_sync(str(src_b))]

    class FakeImages:
        def list(self, filters=None):
            return ["present"]

        def pull(self, image):
            raise AssertionError("image pull should not be needed in this test")

    class FakeContainer:
        def wait(self):
            return {"StatusCode": 0}

        def logs(self):
            return b""

        def remove(self):
            return None

    class FakeContainers:
        def __init__(self):
            self.last_run = None

        def run(self, uri, command=None, **kwargs):
            self.last_run = {"uri": uri, "command": command, "kwargs": kwargs}
            return FakeContainer()

    class FakeClient:
        def __init__(self):
            self.images = FakeImages()
            self.containers = FakeContainers()

    fake_client = FakeClient()

    class FakeDockerModule:
        @staticmethod
        def from_env():
            return fake_client

    monkeypatch.setitem(sys.modules, "docker", FakeDockerModule)

    task = ContainerTask(
        name="test_list_mount",
        image="alpine:latest",
        command=["sh", "-c", "true"],
        inputs={"parts": list[File]},
        outputs={},
    )

    async def fake_get_output(output_directory):
        return ()

    monkeypatch.setattr(task, "_get_output", fake_get_output)

    asyncio.run(task.execute(parts=parts))

    volumes = fake_client.containers.last_run["kwargs"]["volumes"]
    local_dir = next(
        host_path
        for host_path, binding in volumes.items()
        if binding["bind"] == "/var/inputs/parts"
    )

    staged = pathlib.Path(local_dir)
    assert (staged / "0").read_text() == "alpha\n"
    assert (staged / "1").read_text() == "beta\n"


def test_render_command_lowercases_bool_template_inputs():
    task = ContainerTask(
        name="test_bool_render",
        image="alpine:latest",
        command=["echo", "{{.inputs.verbose}}", "{{.inputs.quiet}}"],
        inputs={"verbose": bool, "quiet": bool},
        outputs={},
    )

    commands, _ = task._prepare_command_and_volumes(["{{.inputs.verbose}}", "{{.inputs.quiet}}"], verbose=True, quiet=False)

    assert commands == ["true", "false"]
