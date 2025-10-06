import pytest

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
