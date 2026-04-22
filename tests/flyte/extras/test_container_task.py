import pytest

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
