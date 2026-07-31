import pathlib

import pytest

import flyte
from flyte.extras import ContainerTask
from flyte.io import File


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


def _staged_dir(volume_bindings, bind):
    """Return the host directory bound to `bind` from a volume-bindings map."""
    return pathlib.Path(next(host for host, b in volume_bindings.items() if b["bind"] == bind))


def test_local_execute_materializes_list_of_files(tmp_path):
    flyte.init()
    src_a = tmp_path / "a.txt"
    src_b = tmp_path / "b.txt"
    src_a.write_text("alpha\n")
    src_b.write_text("beta\n")
    parts = [File.from_local_sync(str(src_a)), File.from_local_sync(str(src_b))]

    task = ContainerTask(
        name="test_list_mount",
        image="alpine:latest",
        command=["sh", "-c", "true"],
        inputs={"parts": list[File]},
        outputs={},
    )

    _, volume_bindings = task._prepare_execution_volumes(tmp_path / "outputs", parts=parts)

    # Under the default DIRECT layout, list[File] elements stage at bare indices
    # (0, 1, ...), mirroring how CoPilot stages them remotely.
    staged = _staged_dir(volume_bindings, "/var/inputs/parts")
    assert (staged / "0").read_text() == "alpha\n"
    assert (staged / "1").read_text() == "beta\n"


def test_local_execute_preserves_list_file_names_and_extensions(tmp_path):
    flyte.init()
    src = tmp_path / "reads_1.fastq.gz"
    src.write_text("@r\nACGT\n+\nFFFF\n")
    parts = [File.from_local_sync(str(src))]

    task = ContainerTask(
        name="test_ext",
        image="alpine:latest",
        command=["sh", "-c", "true"],
        inputs={"reads": list[File]},
        outputs={},
        file_input_layout="NAMED_DIR",
    )

    _, volume_bindings = task._prepare_execution_volumes(tmp_path / "outputs", reads=parts)

    staged = _staged_dir(volume_bindings, "/var/inputs/reads")
    # Under NAMED_DIR the original name+extension is preserved so a tool that
    # sniffs format by extension (salmon, etc.) accepts the staged file.
    assert (staged / "reads_1.fastq.gz").exists()


def test_local_execute_dedupes_list_file_name_collisions(tmp_path):
    flyte.init()
    a = tmp_path / "d1" / "reads.fastq.gz"
    b = tmp_path / "d2" / "reads.fastq.gz"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text("a\n")
    b.write_text("b\n")
    parts = [File.from_local_sync(str(a)), File.from_local_sync(str(b))]

    task = ContainerTask(
        name="test_collision",
        image="alpine:latest",
        command=["sh", "-c", "true"],
        inputs={"reads": list[File]},
        outputs={},
        file_input_layout="NAMED_DIR",
    )

    _, volume_bindings = task._prepare_execution_volumes(tmp_path / "outputs", reads=parts)

    staged = _staged_dir(volume_bindings, "/var/inputs/reads")
    names = sorted(p.name for p in staged.iterdir())
    # Both kept (deduped by index prefix), and both end in the real extension.
    assert names == ["1_reads.fastq.gz", "reads.fastq.gz"]
    assert all(n.endswith(".fastq.gz") for n in names)


def test_local_execute_dedupes_when_prefixed_name_collides(tmp_path):
    # A disambiguated name can coincide with another input's real basename:
    # file.txt, 1_file.txt, file.txt. The second file.txt must not clobber the
    # real 1_file.txt; every input has to land under a distinct name.
    flyte.init()
    paths = []
    for i, base in enumerate(("file.txt", "1_file.txt", "file.txt")):
        p = tmp_path / f"d{i}" / base
        p.parent.mkdir()
        p.write_text(f"{i}\n")
        paths.append(p)
    parts = [File.from_local_sync(str(p)) for p in paths]

    task = ContainerTask(
        name="test_prefixed_collision",
        image="alpine:latest",
        command=["sh", "-c", "true"],
        inputs={"reads": list[File]},
        outputs={},
        file_input_layout="NAMED_DIR",
    )

    _, volume_bindings = task._prepare_execution_volumes(tmp_path / "outputs", reads=parts)

    staged = _staged_dir(volume_bindings, "/var/inputs/reads")
    names = sorted(p.name for p in staged.iterdir())
    # Nothing overwritten: all three inputs survive under distinct names.
    assert len(names) == 3
    assert names == ["1_file.txt", "2_file.txt", "file.txt"]


def test_shell_single_file_staged_into_dir_with_original_name(tmp_path):
    from flyte.extras import shell

    flyte.init()
    src = tmp_path / "genome.fasta"
    src.write_text(">x\nACGT\n")
    fasta = File.from_local_sync(str(src))

    task = shell.create(
        name="s",
        image="alpine:latest",
        inputs={"fasta": File},
        outputs={},
        script="cat {inputs.fasta}\n",
    ).as_task()

    commands, volume_bindings = task._prepare_execution_volumes(tmp_path / "outputs", fasta=fasta)

    # A single File is staged into a per-input directory under its original
    # name (so its extension survives), and the command globs that directory.
    staged = _staged_dir(volume_bindings, "/var/inputs/fasta")
    assert (staged / "genome.fasta").read_text() == ">x\nACGT\n"
    assert "/var/inputs/fasta/*" in commands[2]


def test_data_loading_config_direct_leaves_layout_unset():
    from flyteidl2.core import tasks_pb2

    task = ContainerTask(
        name="direct",
        image="alpine:latest",
        command=["true"],
        inputs={"a": File},
        outputs={},
    )
    cfg = task.data_loading_config(None)
    # DIRECT is the proto default (0); the field stays unset so existing tasks
    # serialize identically on older flyteidl2.
    assert cfg.file_input_layout == tasks_pb2.DataLoadingConfig.DIRECT


def test_data_loading_config_shell_emits_named_dir():
    from flyteidl2.core import tasks_pb2

    from flyte.extras import shell

    if not hasattr(tasks_pb2.DataLoadingConfig, "NAMED_DIR"):
        pytest.skip("installed flyteidl2 lacks DataLoadingConfig.file_input_layout")

    task = shell.create(
        name="s",
        image="alpine:latest",
        inputs={"fasta": File},
        outputs={},
        script="cat {inputs.fasta}\n",
    ).as_task()
    cfg = task.data_loading_config(None)
    assert cfg.file_input_layout == tasks_pb2.DataLoadingConfig.NAMED_DIR


def test_render_command_lowercases_bool_template_inputs():
    task = ContainerTask(
        name="test_bool_render",
        image="alpine:latest",
        command=["echo", "{{.inputs.verbose}}", "{{.inputs.quiet}}"],
        inputs={"verbose": bool, "quiet": bool},
        outputs={},
    )

    commands, _ = task._prepare_command_and_volumes(
        ["{{.inputs.verbose}}", "{{.inputs.quiet}}"], verbose=True, quiet=False
    )

    assert commands == ["true", "false"]
