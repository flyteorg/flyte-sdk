from flyte.app import RunOutput


def test_run_output_with_run_name():
    ro = RunOutput(type="string", run_name="my-run-123")
    assert ro.run_name == "my-run-123"
    assert ro.task_name is None
    assert ro.task_version is None
    assert ro.task_auto_version is None
    assert ro.getter == (0,)


def test_run_output_with_task_name():
    ro = RunOutput(type="file", task_name="env.my_task")
    assert ro.task_name == "env.my_task"
    assert ro.run_name is None


def test_run_output_with_task_name_and_version():
    ro = RunOutput(type="directory", task_name="env.my_task", task_version="v1.0")
    assert ro.task_name == "env.my_task"
    assert ro.task_version == "v1.0"


def test_run_output_with_auto_version_latest():
    ro = RunOutput(type="file", task_name="env.my_task", task_auto_version="latest")
    assert ro.task_auto_version == "latest"


def test_run_output_with_auto_version_current():
    ro = RunOutput(type="file", task_name="env.my_task", task_auto_version="current")
    assert ro.task_auto_version == "current"


def test_run_output_custom_getter():
    ro = RunOutput(type="string", run_name="run-1", getter=("output_key", 0, "nested"))
    assert ro.getter == ("output_key", 0, "nested")


def test_run_output_no_run_or_task_creates_instance():
    # RunOutput is a Pydantic model; __post_init__ validation is not called
    # Validation happens at materialize() time instead
    ro = RunOutput(type="string")
    assert ro.run_name is None
    assert ro.task_name is None


def test_run_output_both_run_and_task_creates_instance():
    ro = RunOutput(type="string", run_name="run-1", task_name="task-1")
    assert ro.run_name == "run-1"
    assert ro.task_name == "task-1"


def test_run_output_app_endpoint_type_creates_instance():
    ro = RunOutput(type="app_endpoint", run_name="run-1")
    assert ro.type == "app_endpoint"


def test_run_output_type_mapping_str():
    ro = RunOutput(type=str, run_name="run-1")
    assert ro.type == "string"


def test_run_output_json_roundtrip():
    ro = RunOutput(type="file", task_name="env.my_task", task_auto_version="latest", getter=(0, "key"))
    json_str = ro.model_dump_json()
    restored = RunOutput.model_validate_json(json_str)
    assert restored.task_name == "env.my_task"
    assert restored.task_auto_version == "latest"
    assert restored.getter == (0, "key")
