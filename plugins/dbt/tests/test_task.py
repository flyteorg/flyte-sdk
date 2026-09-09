import sys
import types
from unittest.mock import Mock, patch

from flyte.models import SerializationContext

from flyteplugins.dbt import DbtNodeResult, DbtTask, DbtTaskResolver
from flyteplugins.dbt.runner import _make_on_event_callback, _traced_dbt_node_status, invoke_dbt, on_event


def custom_dbt_callback(event):
    return None


def test_dbt_task_has_cli_args_input():
    task = DbtTask(name="dbt-test")

    assert task.task_type == "dbt"
    assert task.interface.inputs == {"cli_args": (list[str], task.interface.inputs["cli_args"][1])}
    assert task.interface.outputs == {"results": list[DbtNodeResult]}
    assert task.custom_config(SerializationContext(version="v1")) == {}


def test_dbt_task_container_args_include_resolver():
    task = DbtTask(name="dbt-test")

    args = task.container_args(SerializationContext(version="v1"))

    assert "--resolver" in args
    resolver_index = args.index("--resolver")
    assert args[resolver_index + 1] == "flyteplugins.dbt.resolver.DbtTaskResolver"
    assert args[-6:] == [
        "name",
        "dbt-test",
        "include_default_callback",
        "true",
        "callbacks",
        "",
    ]


def test_dbt_task_resolver_round_trips_task():
    task = DbtTask(
        name="dbt-test",
        callbacks=[custom_dbt_callback],
        include_default_callback=False,
    )
    resolver = DbtTaskResolver()

    reconstructed = resolver.load_task(resolver.loader_args(task))

    assert isinstance(reconstructed, DbtTask)
    assert reconstructed.name == "dbt-test"
    assert "cli_args" in reconstructed.interface.inputs
    assert reconstructed.callbacks == ["test_task.custom_dbt_callback"]
    assert reconstructed.include_default_callback is False


def test_dbt_task_forward_invokes_once():
    task = DbtTask(name="dbt-test")
    node_result = DbtNodeResult(
        unique_id="test.project.not_null_orders_order_id.abc",
        name="not_null_orders_order_id",
        resource_type="test",
        status="pass",
    )

    with patch("flyteplugins.dbt.task.invoke_dbt", return_value=[node_result]) as p:
        result = task(cli_args=["test", "--quiet"])

    assert result == [node_result]
    p.assert_called_once_with(
        ["test", "--quiet"],
        callbacks=[],
        include_default_callback=True,
    )


def test_dbt_task_forward_passes_custom_callbacks():
    task = DbtTask(
        name="dbt-test",
        callbacks=[custom_dbt_callback],
        include_default_callback=False,
    )

    with patch("flyteplugins.dbt.task.invoke_dbt", return_value=[]) as p:
        result = task(cli_args=["test", "--quiet"])

    assert result == []
    p.assert_called_once_with(
        ["test", "--quiet"],
        callbacks=[custom_dbt_callback],
        include_default_callback=False,
    )


def test_invoke_dbt_summarizes_runner_result():
    node = Mock(name="node")
    node.name = "not_null_orders_order_id"
    node.resource_type = "test"
    node.unique_id = "test.project.not_null_orders_order_id.abc"

    raw_result = Mock()
    raw_result.unique_id = node.unique_id
    raw_result.node = node
    raw_result.status = "pass"
    raw_result.message = None
    raw_result.failures = 0
    raw_result.execution_time = 1.25
    raw_result.relation_name = None

    runner = Mock()
    runner.invoke.return_value = Mock(success=True, result=[raw_result], exception=None)

    dbt_module = types.ModuleType("dbt")
    dbt_cli_module = types.ModuleType("dbt.cli")
    dbt_cli_main_module = types.ModuleType("dbt.cli.main")
    dbt_cli_main_module.dbtRunner = Mock(return_value=runner)

    with patch.dict(
        sys.modules,
        {
            "dbt": dbt_module,
            "dbt.cli": dbt_cli_module,
            "dbt.cli.main": dbt_cli_main_module,
        },
    ):
        result = invoke_dbt(["test", "--quiet"])

    runner.invoke.assert_called_once_with(["test", "--quiet"])
    dbt_cli_main_module.dbtRunner.assert_called_once()
    callbacks = dbt_cli_main_module.dbtRunner.call_args.kwargs["callbacks"]
    assert len(callbacks) == 1
    assert callbacks[0].__name__ == "wrapped_callback"
    assert len(result) == 1
    assert result[0].name == "not_null_orders_order_id"
    assert result[0].status == "pass"


def test_invoke_dbt_passes_custom_callbacks_to_runner():
    runner = Mock()
    runner.invoke.return_value = Mock(success=True, result=[], exception=None)

    dbt_module = types.ModuleType("dbt")
    dbt_cli_module = types.ModuleType("dbt.cli")
    dbt_cli_main_module = types.ModuleType("dbt.cli.main")
    dbt_cli_main_module.dbtRunner = Mock(return_value=runner)

    with patch.dict(
        sys.modules,
        {
            "dbt": dbt_module,
            "dbt.cli": dbt_cli_module,
            "dbt.cli.main": dbt_cli_main_module,
        },
    ):
        result = invoke_dbt(
            ["test", "--quiet"],
            callbacks=[custom_dbt_callback],
            include_default_callback=False,
        )

    assert result == []
    callbacks = dbt_cli_main_module.dbtRunner.call_args.kwargs["callbacks"]
    assert len(callbacks) == 1

    event = Mock()
    callbacks[0](event)


def test_invoke_dbt_summarizes_wrapped_runner_result():
    node = Mock(name="node")
    node.name = "not_null_orders_order_id"
    node.resource_type = "test"
    node.unique_id = "test.project.not_null_orders_order_id.abc"

    raw_result = Mock()
    raw_result.unique_id = node.unique_id
    raw_result.node = node
    raw_result.status = "pass"
    raw_result.message = None
    raw_result.failures = 0
    raw_result.execution_time = 1.25
    raw_result.relation_name = None

    runner = Mock()
    runner.invoke.return_value = Mock(success=True, result=Mock(results=[raw_result]), exception=None)

    dbt_module = types.ModuleType("dbt")
    dbt_cli_module = types.ModuleType("dbt.cli")
    dbt_cli_main_module = types.ModuleType("dbt.cli.main")
    dbt_cli_main_module.dbtRunner = Mock(return_value=runner)

    with patch.dict(
        sys.modules,
        {
            "dbt": dbt_module,
            "dbt.cli": dbt_cli_module,
            "dbt.cli.main": dbt_cli_main_module,
        },
    ):
        result = invoke_dbt(["test", "--quiet"])

    assert len(result) == 1
    assert result[0].name == "not_null_orders_order_id"
    assert result[0].status == "pass"


def test_invoke_dbt_raises_runner_exception_on_failure():
    runner = Mock()
    runner.invoke.return_value = Mock(success=False, result=[], exception=ValueError("bad dbt"))

    dbt_module = types.ModuleType("dbt")
    dbt_cli_module = types.ModuleType("dbt.cli")
    dbt_cli_main_module = types.ModuleType("dbt.cli.main")
    dbt_cli_main_module.dbtRunner = Mock(return_value=runner)

    with patch.dict(
        sys.modules,
        {
            "dbt": dbt_module,
            "dbt.cli": dbt_cli_module,
            "dbt.cli.main": dbt_cli_main_module,
        },
    ):
        try:
            invoke_dbt(["test", "--quiet"])
        except ValueError as exc:
            assert str(exc) == "bad dbt"
        else:
            raise AssertionError("Expected dbt exception to be raised")


def test_on_event_traces_node_name_to_status():
    event = Mock()
    event.info.name = "NodeFinished"
    event.data.status = "pass"
    event.data.node_info = {
        "node_name": "not_null_orders_order_id",
        "node_status": "running",
    }

    with patch("flyteplugins.dbt.runner._record_dbt_node_status", return_value="pass") as p:
        on_event(event)

    p.assert_called_once_with(
        "not_null_orders_order_id",
        "running",
        trace_name="running.not_null_orders_order_id",
    )


def test_on_event_falls_back_to_event_status():
    event = Mock()
    event.info.name = "NodeFinished"
    event.data.status = "pass"
    event.data.node_info = {"node_name": "not_null_orders_order_id"}

    with patch("flyteplugins.dbt.runner._record_dbt_node_status", return_value="pass") as p:
        on_event(event)

    p.assert_called_once_with(
        "not_null_orders_order_id",
        "pass",
        trace_name="pass.not_null_orders_order_id",
    )


def test_on_event_ignores_non_node_events():
    event = Mock()
    event.info.name = "LogStartLine"
    event.data.status = "pass"
    event.data.node_info = {
        "node_name": "not_null_orders_order_id",
        "node_status": "running",
    }

    with patch("flyteplugins.dbt.runner._record_dbt_node_status") as p:
        on_event(event)

    p.assert_not_called()


def test_on_event_callback_reenters_flyte_context():
    event = Mock()

    with patch("flyteplugins.dbt.runner.on_event") as p:
        callback = _make_on_event_callback()
        callback(event)

    p.assert_called_once_with(event)


def test_traced_dbt_node_status_uses_trace_name():
    captured = {}

    def fake_trace(func):
        captured["name"] = func.__name__
        captured["qualname"] = func.__qualname__

        def wrapper(node_name):
            return func(node_name)

        return wrapper

    with patch("flyte.trace", side_effect=fake_trace):
        _traced_dbt_node_status(
            "not_null_orders_order_id",
            trace_name="NodeFinished.not_null_orders_order_id",
        )

    assert captured == {
        "name": "NodeFinished.not_null_orders_order_id",
        "qualname": "NodeFinished.not_null_orders_order_id",
    }
