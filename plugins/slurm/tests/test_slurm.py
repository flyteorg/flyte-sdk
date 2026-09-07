import json
import pathlib

import pytest
from flyte.connectors import ConnectorRegistry
from flyte.models import SerializationContext
from flyteidl2.core import tasks_pb2
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.core.literals_pb2 import KeyValuePair
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

from flyteplugins.slurm.connector import SlurmConnector, SlurmJobMetadata, slurm_state_to_phase
from flyteplugins.slurm.script import pyxis_image_ref, render_container_job, render_script_job
from flyteplugins.slurm.task import Slurm, SlurmFunctionTask, SlurmScriptTask
from flyteplugins.slurm.transport import SlurmJobState, parse_sacct, parse_sbatch_job_id, parse_squeue


@pytest.fixture
def sctx() -> SerializationContext:
    return SerializationContext(
        project="p",
        domain="d",
        version="v",
        org="o",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )


class TestImageRef:
    @pytest.mark.parametrize(
        "image, expected",
        [
            ("ghcr.io/flyteorg/flyte:py3.12-v2", "ghcr.io#flyteorg/flyte:py3.12-v2"),
            ("cr.eu-north1.nebius.cloud/org/img:tag", "cr.eu-north1.nebius.cloud#org/img:tag"),
            ("localhost:5000/img:tag", "localhost:5000#img:tag"),
            ("python:3.12-slim", "python:3.12-slim"),
            ("library/python:3.12", "library/python:3.12"),
            ("docker://ghcr.io/org/img:tag", "ghcr.io#org/img:tag"),
            ("ghcr.io#org/img:tag", "ghcr.io#org/img:tag"),
            ("/jail/images/train.sqsh", "/jail/images/train.sqsh"),
        ],
    )
    def test_conversion(self, image, expected):
        assert pyxis_image_ref(image) == expected


class TestScriptRendering:
    def test_container_job(self):
        script = render_container_job(
            job_name="flyte-train-abc",
            stdout_path="/home/u/.flyte/jobs/flyte-train-abc.out",
            stderr_path="/home/u/.flyte/jobs/flyte-train-abc.err",
            image="ghcr.io/org/train:1",
            command=["a0", "--inputs", "s3://b/in.pb", "--name", "a b"],
            env={"B": "2", "A": "it's"},
            sbatch_fields={"partition": "main", "nodes": 2, "gres": "gpu:8", "time_limit": "1:00:00"},
            sbatch_extra={"exclusive": True, "partition": "override"},
            container_mounts=["/data:/data"],
            container_workdir="/",
        )
        lines = script.splitlines()
        assert lines[0] == "#!/bin/bash"
        assert "#SBATCH --job-name=flyte-train-abc" in lines
        assert "#SBATCH --output=/home/u/.flyte/jobs/flyte-train-abc.out" in lines
        assert "#SBATCH --nodes=2" in lines
        assert "#SBATCH --gres=gpu:8" in lines
        assert "#SBATCH --time=1:00:00" in lines
        assert "#SBATCH --exclusive" in lines
        # passthrough wins over the first-class field
        assert "#SBATCH --partition=override" in lines
        assert "#SBATCH --partition=main" not in lines
        # env is exported sorted and quoted
        assert lines.index("export A='it'\"'\"'s'") < lines.index("export B=2")
        assert "set -euo pipefail" in lines
        srun = lines[-1]
        # '#' is shell-special, so the image ref is quoted; still a valid srun line
        assert srun.startswith("srun '--container-image=ghcr.io#org/train:1' --container-mounts=/data:/data")
        assert "--container-workdir=/" in srun
        assert srun.endswith("a0 --inputs s3://b/in.pb --name 'a b'")

    def test_container_job_requires_command(self):
        with pytest.raises(ValueError):
            render_container_job(
                job_name="j", stdout_path="o", stderr_path="e", image="x", command=[], env={}, sbatch_fields={}
            )

    def test_rejects_bad_option_names(self):
        with pytest.raises(ValueError):
            render_container_job(
                job_name="j",
                stdout_path="o",
                stderr_path="e",
                image="x",
                command=["a0"],
                env={},
                sbatch_fields={},
                sbatch_extra={"bad option; rm -rf": "1"},
            )

    def test_script_job_keeps_user_script_and_drops_shebang(self):
        user = "#!/bin/bash\n#SBATCH --time=9:00:00\necho hello\nsrun ./train.sh\n"
        script = render_script_job(
            job_name="flyte-script-1",
            stdout_path="/h/o.out",
            stderr_path="/h/o.err",
            script=user,
            env={"FLYTE_INPUT_EPOCHS": "3"},
            sbatch_fields={"partition": "main"},
        )
        assert script.startswith("#!/bin/bash\n#SBATCH --job-name=flyte-script-1")
        assert script.count("#!/bin/bash") == 1
        assert "export FLYTE_INPUT_EPOCHS=3" in script
        assert "echo hello\nsrun ./train.sh\n" in script
        # user's own directive is preserved (Slurm treats it as a comment after ours)
        assert "#SBATCH --time=9:00:00" in script


class TestParsing:
    def test_sacct(self):
        out = (
            "123|COMPLETED|0:0|None\n"
            "123.batch|COMPLETED|0:0|\n"
            "124|CANCELLED by 1000|0:0|None\n"
            "125|FAILED|1:0|NonZeroExitCode\n"
        )
        states = parse_sacct(out)
        assert set(states) == {"123", "124", "125"}
        assert states["123"].exit_code == "0:0" and states["123"].reason is None
        assert states["124"].base_state == "CANCELLED"
        assert states["125"].reason == "NonZeroExitCode"

    def test_squeue(self):
        states = parse_squeue("200|PENDING|Priority\n201|RUNNING|None\n")
        assert states["200"].state == "PENDING" and states["200"].reason == "Priority"
        assert states["201"].reason is None

    def test_sbatch_parsable(self):
        assert parse_sbatch_job_id("42\n") == "42"
        assert parse_sbatch_job_id("42;cluster1\n") == "42"
        with pytest.raises(RuntimeError):
            parse_sbatch_job_id("sbatch: error: invalid partition")


class TestPhaseMapping:
    @pytest.mark.parametrize(
        "state, phase",
        [
            ("PENDING", TaskExecution.QUEUED),
            ("CONFIGURING", TaskExecution.QUEUED),
            ("RUNNING", TaskExecution.RUNNING),
            ("COMPLETING", TaskExecution.RUNNING),
            ("COMPLETED", TaskExecution.SUCCEEDED),
            ("FAILED", TaskExecution.FAILED),
            ("TIMEOUT", TaskExecution.FAILED),
            ("OUT_OF_MEMORY", TaskExecution.FAILED),
            ("NODE_FAIL", TaskExecution.FAILED),
            ("PREEMPTED", TaskExecution.RETRYABLE_FAILED),
            ("CANCELLED by 1000", TaskExecution.ABORTED),
            ("SOMETHING_NEW", TaskExecution.RUNNING),
        ],
    )
    def test_mapping(self, state, phase):
        assert slurm_state_to_phase(state) == phase


class TestMetadata:
    def test_round_trip(self):
        meta = SlurmJobMetadata(
            job_id="7",
            job_name="flyte-x-1",
            host="login",
            username="u",
            port=2222,
            stdout_path="/h/x.out",
            stderr_path="/h/x.err",
            known_hosts="/etc/kh",
        )
        decoded = SlurmJobMetadata.decode(meta.encode())
        assert decoded == meta
        assert json.loads(meta.encode())["port"] == 2222


class TestTaskConfig:
    def test_custom_config(self, sctx):
        cfg = Slurm(
            partition="main",
            nodes=4,
            gres="gpu:8",
            time_limit="4:00:00",
            sbatch_options={"exclusive": True},
            container_mounts=["/data:/data"],
            env={"AWS_REGION": "eu-north1"},
            host="login.example",
            username="flyte",
            ssh_private_key="slurm-key",
        )
        task = SlurmFunctionTask(plugin_config=cfg, name="train", interface=None, func=lambda: None)
        assert task.task_type == "slurm"
        custom = task.custom_config(sctx)
        assert custom["sbatch"] == {"partition": "main", "nodes": 4, "gres": "gpu:8", "time_limit": "4:00:00"}
        assert custom["sbatch_options"] == {"exclusive": True}
        assert custom["connection"] == {"host": "login.example", "username": "flyte"}
        assert custom["container"]["mounts"] == ["/data:/data"]
        assert custom["env"] == {"AWS_REGION": "eu-north1"}
        assert custom["secrets"] == {"ssh_private_key": "slurm-key"}

    def test_custom_config_without_connection_uses_connector_defaults(self, sctx):
        task = SlurmFunctionTask(plugin_config=Slurm(partition="main"), name="t", interface=None, func=lambda: None)
        custom = task.custom_config(sctx)
        assert custom["connection"] == {}
        assert "secrets" not in custom

    def test_script_task(self, sctx):
        task = SlurmScriptTask(
            name="legacy",
            script="#!/bin/bash\nsrun ./train.sh\n",
            plugin_config=Slurm(partition="main"),
            inputs={"epochs": int},
        )
        assert task.task_type == "slurm_script"
        assert task.custom_config(sctx)["script"].endswith("srun ./train.sh\n")
        assert "epochs" in task.native_interface.inputs

    def test_registered_with_plugin_registry(self):
        from flyte._task_plugins import TaskPluginRegistry

        assert TaskPluginRegistry.find(Slurm) is SlurmFunctionTask


class TestConnectorRegistration:
    def test_both_task_types_registered(self):
        assert isinstance(ConnectorRegistry.get_connector("slurm"), SlurmConnector)
        assert isinstance(ConnectorRegistry.get_connector("slurm_script"), SlurmConnector)
        names = {c.name for c in ConnectorRegistry._list_connectors()}
        assert "Slurm Connector" in names


class _FakeTransport:
    def __init__(self):
        self.submitted = []
        self.cancelled = []
        self.states = {}
        self.tails = {}

    async def home(self):
        return "/home/flyte"

    async def submit(self, script, script_path):
        self.submitted.append((script, script_path))
        return "900"

    async def status(self, job_ids):
        return {j: self.states[j] for j in job_ids if j in self.states}

    async def cancel(self, job_id):
        self.cancelled.append(job_id)

    async def tail(self, path, lines=100):
        return self.tails.get(path, "")


def _task_template(task_type: str, custom: dict, image="ghcr.io/org/train:1") -> tasks_pb2.TaskTemplate:
    struct = Struct()
    json_format.ParseDict(custom, struct)
    tt = tasks_pb2.TaskTemplate(type=task_type, custom=struct)
    tt.id.name = "pkg.mod.train"
    tt.container.image = image
    tt.container.args.extend(["a0", "--inputs", "s3://b/in.pb", "--outputs-path", "s3://b/out"])
    tt.container.env.append(KeyValuePair(key="FROM_TEMPLATE", value="1"))
    return tt


@pytest.mark.asyncio
class TestConnector:
    async def test_create_native(self, monkeypatch):
        connector = SlurmConnector()
        fake = _FakeTransport()
        monkeypatch.setattr(connector, "_transport", lambda *a, **k: fake)
        custom = Slurm(partition="main", nodes=2, gres="gpu:8", host="login", username="flyte").to_custom_config()
        custom["env"] = {"X": "y"}

        meta = await connector.create(_task_template("slurm", custom), "s3://b/out", ssh_private_key="KEY")

        assert meta.job_id == "900"
        assert meta.host == "login" and meta.username == "flyte" and meta.port == 22
        assert meta.stdout_path == f"/home/flyte/.flyte/jobs/{meta.job_name}.out"
        script, path = fake.submitted[0]
        assert path == f"/home/flyte/.flyte/jobs/{meta.job_name}.sbatch"
        # Struct round-trip turned 2 into 2.0; make sure the directive is an int
        assert "#SBATCH --nodes=2\n" in script
        assert "#SBATCH --gres=gpu:8" in script
        assert "export FROM_TEMPLATE=1" in script and "export X=y" in script
        assert "srun '--container-image=ghcr.io#org/train:1' a0 --inputs s3://b/in.pb" in script

    async def test_create_script_exposes_inputs(self, monkeypatch):
        connector = SlurmConnector()
        fake = _FakeTransport()
        monkeypatch.setattr(connector, "_transport", lambda *a, **k: fake)
        custom = Slurm(partition="main", host="login", username="flyte", working_dir="/scratch/jobs").to_custom_config()
        custom["script"] = "#!/bin/bash\nsrun ./train.sh $FLYTE_INPUT_EPOCHS\n"

        meta = await connector.create(
            _task_template("slurm_script", custom),
            "s3://b/out",
            inputs={"epochs": 3, "name": "x"},
            ssh_private_key="KEY",
        )

        script, path = fake.submitted[0]
        assert path.startswith("/scratch/jobs/")
        assert meta.stderr_path == f"/scratch/jobs/{meta.job_name}.err"
        assert "export FLYTE_INPUT_EPOCHS=3" in script
        assert "export FLYTE_INPUT_NAME=x" in script
        assert "srun ./train.sh $FLYTE_INPUT_EPOCHS" in script
        assert "--container-image" not in script

    async def test_create_requires_connection(self, monkeypatch):
        connector = SlurmConnector()
        monkeypatch.delenv("FLYTE_SLURM_HOST", raising=False)
        monkeypatch.delenv("FLYTE_SLURM_USERNAME", raising=False)
        with pytest.raises(ValueError, match="Missing Slurm connection"):
            await connector.create(_task_template("slurm", Slurm().to_custom_config()), "s3://b", ssh_private_key="KEY")

    async def test_create_uses_connector_env_defaults(self, monkeypatch):
        connector = SlurmConnector()
        fake = _FakeTransport()
        captured = {}

        def _transport(host, port, username, private_key, known_hosts, skip):
            captured.update(host=host, port=port, username=username, key=private_key)
            return fake

        monkeypatch.setattr(connector, "_transport", _transport)
        monkeypatch.setenv("FLYTE_SLURM_HOST", "env-login")
        monkeypatch.setenv("FLYTE_SLURM_USERNAME", "env-user")
        monkeypatch.setenv("FLYTE_SLURM_PORT", "2222")
        monkeypatch.setenv("FLYTE_SLURM_SSH_PRIVATE_KEY", "ENVKEY")

        meta = await connector.create(_task_template("slurm", Slurm(partition="p").to_custom_config()), "s3://b")
        assert (meta.host, meta.username, meta.port) == ("env-login", "env-user", 2222)
        assert captured["key"] == "ENVKEY"

    async def test_get_maps_state_and_surfaces_stderr(self, monkeypatch):
        connector = SlurmConnector()
        fake = _FakeTransport()
        monkeypatch.setattr(connector, "_transport", lambda *a, **k: fake)
        meta = SlurmJobMetadata("900", "flyte-t-1", "login", "flyte", 22, "/h/t.out", "/h/t.err")

        fake.states["900"] = SlurmJobState("900", "PENDING", reason="Priority")
        res = await connector.get(meta, ssh_private_key="KEY")
        assert res.phase == TaskExecution.QUEUED
        assert "Priority" in res.message
        assert [link.name for link in res.log_links] == ["Slurm stdout", "Slurm stderr"]

        fake.states["900"] = SlurmJobState("900", "FAILED", exit_code="1:0", reason="NonZeroExitCode")
        fake.tails["/h/t.err"] = "Traceback...\nValueError: boom\n"
        res = await connector.get(meta, ssh_private_key="KEY")
        assert res.phase == TaskExecution.FAILED
        assert "exit code 1:0" in res.message and "ValueError: boom" in res.message

        fake.states["900"] = SlurmJobState("900", "COMPLETED", exit_code="0:0")
        res = await connector.get(meta, ssh_private_key="KEY")
        assert res.phase == TaskExecution.SUCCEEDED

    async def test_get_unknown_job_raises(self, monkeypatch):
        connector = SlurmConnector()
        monkeypatch.setattr(connector, "_transport", lambda *a, **k: _FakeTransport())
        meta = SlurmJobMetadata("1", "n", "login", "flyte", 22, "/o", "/e")
        with pytest.raises(RuntimeError, match="not known"):
            await connector.get(meta, ssh_private_key="KEY")

    async def test_delete_cancels(self, monkeypatch):
        connector = SlurmConnector()
        fake = _FakeTransport()
        monkeypatch.setattr(connector, "_transport", lambda *a, **k: fake)
        meta = SlurmJobMetadata("900", "n", "login", "flyte", 22, "/o", "/e")
        await connector.delete(meta, ssh_private_key="KEY")
        assert fake.cancelled == ["900"]

    async def test_get_logs(self, monkeypatch):
        connector = SlurmConnector()
        fake = _FakeTransport()
        fake.tails["/o"] = "line one\nline two\n"
        monkeypatch.setattr(connector, "_transport", lambda *a, **k: fake)
        meta = SlurmJobMetadata("900", "n", "login", "flyte", 22, "/o", "/e")
        responses = [r async for r in connector.get_logs(meta, ssh_private_key="KEY")]
        assert [line.message for line in responses[0].body.lines] == ["line one", "line two"]

    async def test_missing_key_is_a_clear_error(self, monkeypatch):
        monkeypatch.delenv("FLYTE_SLURM_SSH_PRIVATE_KEY", raising=False)
        meta = SlurmJobMetadata("900", "n", "login", "flyte", 22, "/o", "/e")
        with pytest.raises(ValueError, match="SSH private key"):
            await SlurmConnector().get(meta)
