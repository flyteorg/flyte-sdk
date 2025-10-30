from datetime import datetime

import pytest
from flyteidl2.core import interface_pb2, literals_pb2, types_pb2
from flyteidl2.task import common_pb2

from flyte import Cron, FixedRate, TaskEnvironment, Trigger, TriggerTime
from flyte._internal.runtime.convert import convert_upload_default_inputs
from flyte._internal.runtime.trigger_serde import (
    _to_schedule,
    process_default_inputs,
    to_task_trigger,
)
from flyte.types import TypeEngine


class TestToSchedule:
    """Test the _to_schedule function"""

    def test_cron_without_kickoff_arg(self):
        """Test Cron schedule without kickoff argument"""
        cron = Cron("0 * * * *")
        schedule = _to_schedule(cron)

        assert schedule.cron_expression == "0 * * * *"
        assert schedule.kickoff_time_input_arg == ""

    def test_cron_with_kickoff_arg(self):
        """Test Cron schedule with kickoff argument"""
        cron = Cron("0 0 * * *")
        schedule = _to_schedule(cron, kickoff_arg_name="trigger_time")

        assert schedule.cron_expression == "0 0 * * *"
        assert schedule.kickoff_time_input_arg == "trigger_time"

    def test_fixed_rate_without_start_time(self):
        """Test FixedRate schedule without start time"""
        fixed_rate = FixedRate(interval_minutes=60)
        schedule = _to_schedule(fixed_rate)

        assert schedule.rate.value == 60
        assert schedule.rate.unit == common_pb2.FixedRateUnit.FIXED_RATE_UNIT_MINUTE
        assert not schedule.rate.HasField("start_time")
        assert schedule.kickoff_time_input_arg == ""

    def test_fixed_rate_with_start_time(self):
        """Test FixedRate schedule with start time"""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        fixed_rate = FixedRate(interval_minutes=30, start_time=start_time)
        schedule = _to_schedule(fixed_rate, kickoff_arg_name="kickoff")

        assert schedule.rate.value == 30
        assert schedule.rate.unit == common_pb2.FixedRateUnit.FIXED_RATE_UNIT_MINUTE
        assert schedule.rate.start_time.ToDatetime() == start_time
        assert schedule.kickoff_time_input_arg == "kickoff"

    def test_cron_expression_variations(self):
        """Test various cron expressions"""
        expressions = [
            "* * * * *",  # every minute
            "0 0 * * 0",  # weekly on Sunday
            "0 0 1 * *",  # monthly on 1st
        ]

        for expr in expressions:
            cron = Cron(expr)
            schedule = _to_schedule(cron)
            assert schedule.cron_expression == expr


class TestProcessDefaultInputs:
    """Test the process_default_inputs function"""

    @pytest.mark.asyncio
    async def test_empty_default_inputs(self):
        """Test with no default inputs"""
        task_inputs = interface_pb2.VariableMap()
        task_default_inputs = []

        result = await process_default_inputs({}, "test_task", task_inputs, task_default_inputs)

        assert result == []

    @pytest.mark.asyncio
    async def test_valid_default_inputs(self):
        """Test with valid default inputs"""
        # Create task inputs with int and string variables
        task_inputs = interface_pb2.VariableMap(
            variables={
                "num": interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER)),
                "text": interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING)),
            }
        )

        default_inputs = {"num": 42, "text": "hello"}

        result = await process_default_inputs(default_inputs, "test_task", task_inputs, [])

        assert len(result) == 2
        assert result[0].name == "num"
        assert result[0].value.scalar.primitive.integer == 42
        assert result[1].name == "text"
        assert result[1].value.scalar.primitive.string_value == "hello"

    @pytest.mark.asyncio
    async def test_invalid_input_name(self):
        """Test with input name not in task inputs"""
        task_inputs = interface_pb2.VariableMap(
            variables={
                "valid_input": interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER))
            }
        )

        default_inputs = {"invalid_input": 42}

        with pytest.raises(ValueError, match="Trigger default input 'invalid_input' must be an input to the task"):
            await process_default_inputs(default_inputs, "test_task", task_inputs, [])

    @pytest.mark.asyncio
    async def test_task_default_inputs_merged(self):
        """Test that task default inputs are merged with trigger defaults"""
        # Create task inputs
        task_inputs = interface_pb2.VariableMap(
            variables={
                "a": interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER)),
                "b": interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING)),
            }
        )

        # Create task default input for 'b'
        task_default_inputs = [
            common_pb2.NamedParameter(
                name="b",
                parameter=interface_pb2.Parameter(
                    default=literals_pb2.Literal(
                        scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(string_value="default_b"))
                    )
                ),
            )
        ]

        # Provide trigger default only for 'a'
        default_inputs = {"a": 10}

        result = await process_default_inputs(default_inputs, "test_task", task_inputs, task_default_inputs)

        assert len(result) == 2
        assert result[0].name == "a"
        assert result[0].value.scalar.primitive.integer == 10
        assert result[1].name == "b"
        assert result[1].value.scalar.primitive.string_value == "default_b"

    @pytest.mark.asyncio
    async def test_trigger_defaults_override_task_defaults(self):
        """Test that trigger defaults override task defaults"""
        task_inputs = interface_pb2.VariableMap(
            variables={
                "x": interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER)),
            }
        )

        task_default_inputs = [
            common_pb2.NamedParameter(
                name="x",
                parameter=interface_pb2.Parameter(
                    default=literals_pb2.Literal(
                        scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=100))
                    )
                ),
            )
        ]

        default_inputs = {"x": 200}

        result = await process_default_inputs(default_inputs, "test_task", task_inputs, task_default_inputs)

        assert len(result) == 1
        assert result[0].name == "x"
        # Trigger default should override task default
        assert result[0].value.scalar.primitive.integer == 200


class TestToTaskTrigger:
    """Test the to_task_trigger function"""

    @pytest.mark.asyncio
    async def test_basic_cron_trigger(self):
        """Test basic Cron trigger conversion"""
        trigger = Trigger(
            name="test_trigger",
            automation=Cron("0 * * * *"),
        )

        task_inputs = interface_pb2.VariableMap()
        task_default_inputs = []

        result = await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

        assert result.name == "test_trigger"
        assert result.spec.active is True
        assert result.automation_spec.type == common_pb2.TriggerAutomationSpecType.TYPE_SCHEDULE
        assert result.automation_spec.schedule.cron_expression == "0 * * * *"

    @pytest.mark.asyncio
    async def test_fixed_rate_trigger(self):
        """Test FixedRate trigger conversion"""
        trigger = Trigger(
            name="fixed_rate_trigger",
            automation=FixedRate(interval_minutes=15),
            auto_activate=False,
        )

        task_inputs = interface_pb2.VariableMap()
        task_default_inputs = []

        result = await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

        assert result.name == "fixed_rate_trigger"
        assert result.spec.active is False
        assert result.automation_spec.schedule.rate.value == 15
        assert result.automation_spec.schedule.rate.unit == common_pb2.FixedRateUnit.FIXED_RATE_UNIT_MINUTE

    @pytest.mark.asyncio
    async def test_trigger_with_env_vars(self):
        """Test trigger with environment variables"""
        trigger = Trigger(
            name="env_trigger",
            automation=Cron("0 0 * * *"),
            env_vars={"KEY1": "value1", "KEY2": "value2"},
        )

        task_inputs = interface_pb2.VariableMap()
        task_default_inputs = []

        result = await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

        assert result.spec.run_spec.envs is not None
        env_dict = {kv.key: kv.value for kv in result.spec.run_spec.envs.values}
        assert env_dict == {"KEY1": "value1", "KEY2": "value2"}

    @pytest.mark.asyncio
    async def test_trigger_with_labels_and_annotations(self):
        """Test trigger with labels and annotations"""
        trigger = Trigger(
            name="labeled_trigger",
            automation=Cron("0 0 * * *"),
            labels={"label1": "value1", "label2": "value2"},
            annotations={"anno1": "value1"},
        )

        task_inputs = interface_pb2.VariableMap()
        task_default_inputs = []

        result = await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

        assert result.spec.run_spec.labels.values == {"label1": "value1", "label2": "value2"}
        assert result.spec.run_spec.annotations.values == {"anno1": "value1"}

    @pytest.mark.asyncio
    async def test_trigger_with_interruptible(self):
        """Test trigger with interruptible flag"""
        trigger = Trigger(
            name="interruptible_trigger",
            automation=Cron("0 0 * * *"),
            interruptible=True,
        )

        task_inputs = interface_pb2.VariableMap()
        task_default_inputs = []

        result = await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

        assert result.spec.run_spec.interruptible.value is True

    @pytest.mark.asyncio
    async def test_trigger_with_overwrite_cache(self):
        """Test trigger with overwrite_cache"""
        trigger = Trigger(
            name="cache_trigger",
            automation=Cron("0 0 * * *"),
            overwrite_cache=True,
        )

        task_inputs = interface_pb2.VariableMap()
        task_default_inputs = []

        result = await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

        assert result.spec.run_spec.overwrite_cache is True

    @pytest.mark.asyncio
    async def test_trigger_with_queue(self):
        """Test trigger with queue"""
        trigger = Trigger(
            name="queue_trigger",
            automation=Cron("0 0 * * *"),
            queue="gpu-queue",
        )

        task_inputs = interface_pb2.VariableMap()
        task_default_inputs = []

        result = await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

        assert result.spec.run_spec.cluster == "gpu-queue"

    @pytest.mark.asyncio
    async def test_trigger_with_trigger_time(self):
        """Test trigger with TriggerTime input"""
        trigger = Trigger(
            name="timed_trigger",
            automation=Cron("0 0 * * *"),
            inputs={"trigger_time": TriggerTime},
        )

        task_inputs = interface_pb2.VariableMap(
            variables={
                "trigger_time": interface_pb2.Variable(
                    type=types_pb2.LiteralType(simple=types_pb2.SimpleType.DATETIME)
                ),
            }
        )
        task_default_inputs = []

        result = await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

        assert result.automation_spec.schedule.kickoff_time_input_arg == "trigger_time"

    @pytest.mark.asyncio
    async def test_trigger_with_trigger_time_invalid_input(self):
        """Test trigger with TriggerTime for non-existent input"""
        trigger = Trigger(
            name="invalid_trigger",
            automation=Cron("0 0 * * *"),
            inputs={"nonexistent": TriggerTime},
        )

        task_inputs = interface_pb2.VariableMap(
            variables={
                "valid_input": interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.DATETIME)),
            }
        )
        task_default_inputs = []

        with pytest.raises(ValueError, match="TriggerTime input 'nonexistent' must be an input to the task"):
            await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

    @pytest.mark.asyncio
    async def test_trigger_with_default_inputs(self):
        """Test trigger with default inputs"""
        trigger = Trigger(
            name="defaults_trigger",
            automation=Cron("0 0 * * *"),
            inputs={"num": 42, "text": "hello"},
        )

        task_inputs = interface_pb2.VariableMap(
            variables={
                "num": interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER)),
                "text": interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING)),
            }
        )
        task_default_inputs = []

        result = await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

        assert len(result.spec.inputs.literals) == 2
        input_dict = {lit.name: lit.value for lit in result.spec.inputs.literals}
        assert input_dict["num"].scalar.primitive.integer == 42
        assert input_dict["text"].scalar.primitive.string_value == "hello"

    @pytest.mark.asyncio
    async def test_trigger_with_mixed_inputs(self):
        """Test trigger with both TriggerTime and default inputs"""
        trigger = Trigger(
            name="mixed_trigger",
            automation=Cron("0 0 * * *"),
            inputs={"trigger_time": TriggerTime, "count": 100},
        )

        task_inputs = interface_pb2.VariableMap(
            variables={
                "trigger_time": interface_pb2.Variable(
                    type=types_pb2.LiteralType(simple=types_pb2.SimpleType.DATETIME)
                ),
                "count": interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER)),
            }
        )
        task_default_inputs = []

        result = await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

        assert result.automation_spec.schedule.kickoff_time_input_arg == "trigger_time"
        assert len(result.spec.inputs.literals) == 1
        assert result.spec.inputs.literals[0].name == "count"
        assert result.spec.inputs.literals[0].value.scalar.primitive.integer == 100

    @pytest.mark.asyncio
    async def test_trigger_all_run_spec_options(self):
        """Test trigger with all RunSpec options"""
        trigger = Trigger(
            name="full_trigger",
            automation=FixedRate(interval_minutes=30),
            auto_activate=False,
            env_vars={"ENV": "prod"},
            interruptible=False,
            overwrite_cache=True,
            queue="default",
            labels={"team": "ml"},
            annotations={"owner": "data-team"},
        )

        task_inputs = interface_pb2.VariableMap()
        task_default_inputs = []

        result = await to_task_trigger(trigger, "test_task", task_inputs, task_default_inputs)

        assert result.spec.active is False
        assert result.spec.run_spec.overwrite_cache is True
        assert result.spec.run_spec.interruptible.value is False
        assert result.spec.run_spec.cluster == "default"
        assert result.spec.run_spec.labels.values == {"team": "ml"}
        assert result.spec.run_spec.annotations.values == {"owner": "data-team"}
        env_dict = {kv.key: kv.value for kv in result.spec.run_spec.envs.values}
        assert env_dict == {"ENV": "prod"}


class TestAutomationSpec:
    """Test automation_spec separately for different trigger types"""

    @pytest.mark.asyncio
    async def test_automation_spec_cron(self):
        """Test automation_spec for Cron trigger"""
        trigger = Trigger(
            name="cron_trigger",
            automation=Cron("*/5 * * * *"),
        )

        task_inputs = interface_pb2.VariableMap()
        result = await to_task_trigger(trigger, "test_task", task_inputs, [])

        assert result.automation_spec.type == common_pb2.TriggerAutomationSpecType.TYPE_SCHEDULE
        assert result.automation_spec.schedule.cron_expression == "*/5 * * * *"
        assert result.automation_spec.schedule.kickoff_time_input_arg == ""

    @pytest.mark.asyncio
    async def test_automation_spec_fixed_rate_no_start(self):
        """Test automation_spec for FixedRate without start time"""
        trigger = Trigger(
            name="rate_trigger",
            automation=FixedRate(interval_minutes=120),
        )

        task_inputs = interface_pb2.VariableMap()
        result = await to_task_trigger(trigger, "test_task", task_inputs, [])

        assert result.automation_spec.type == common_pb2.TriggerAutomationSpecType.TYPE_SCHEDULE
        assert result.automation_spec.schedule.rate.value == 120
        assert result.automation_spec.schedule.rate.unit == common_pb2.FixedRateUnit.FIXED_RATE_UNIT_MINUTE
        assert not result.automation_spec.schedule.rate.HasField("start_time")

    @pytest.mark.asyncio
    async def test_automation_spec_fixed_rate_with_start(self):
        """Test automation_spec for FixedRate with start time"""
        start = datetime(2025, 6, 1, 10, 30, 0)
        trigger = Trigger(
            name="scheduled_rate_trigger",
            automation=FixedRate(interval_minutes=45, start_time=start),
        )

        task_inputs = interface_pb2.VariableMap()
        result = await to_task_trigger(trigger, "test_task", task_inputs, [])

        assert result.automation_spec.type == common_pb2.TriggerAutomationSpecType.TYPE_SCHEDULE
        assert result.automation_spec.schedule.rate.value == 45
        assert result.automation_spec.schedule.rate.start_time.ToDatetime() == start

    @pytest.mark.asyncio
    async def test_automation_spec_with_kickoff_arg(self):
        """Test automation_spec with kickoff time argument"""
        trigger = Trigger(
            name="kickoff_trigger",
            automation=Cron("0 12 * * *"),
            inputs={"scheduled_at": TriggerTime},
        )

        task_inputs = interface_pb2.VariableMap(
            variables={
                "scheduled_at": interface_pb2.Variable(
                    type=types_pb2.LiteralType(simple=types_pb2.SimpleType.DATETIME)
                ),
            }
        )

        result = await to_task_trigger(trigger, "test_task", task_inputs, [])

        assert result.automation_spec.schedule.cron_expression == "0 12 * * *"
        assert result.automation_spec.schedule.kickoff_time_input_arg == "scheduled_at"


@pytest.mark.asyncio
async def test_task_with_trigger_all_options():
    """Test task with trigger that uses all available options"""
    env = TaskEnvironment(name="test_env")

    trigger = Trigger(
        name="comprehensive",
        automation=FixedRate(interval_minutes=30),
        description="A comprehensive trigger",
        auto_activate=False,
        inputs={
            "trigger_time": TriggerTime,
            "batch_size": 100,
            "mode": "staging",
        },
        env_vars={"ENV": "staging"},
        interruptible=True,
        overwrite_cache=True,
        queue="ml-queue",
        labels={"component": "etl", "team": "data"},
        annotations={"version": "v1.0"},
    )

    @env.task(triggers=trigger)
    async def comprehensive_task(
        trigger_time: datetime,
        batch_size: int = 10,
        mode: str = "dev",
        region: str = "us-east",
        learning_rate: float = 0.01,
    ) -> str:
        return "ok"

    # Convert to TaskTrigger proto and validate all fields
    task_template = comprehensive_task

    # Convert native interface inputs to protobuf VariableMap
    task_inputs = interface_pb2.VariableMap()
    for input_name, (input_type, _) in task_template.interface.inputs.items():
        lt = TypeEngine.to_literal_type(input_type)
        task_inputs.variables[input_name].CopyFrom(interface_pb2.Variable(type=lt))

    # Convert default inputs to NamedParameters
    task_default_inputs = await convert_upload_default_inputs(task_template.interface)

    result = await to_task_trigger(trigger, task_template.name, task_inputs, task_default_inputs)

    # Validate trigger metadata
    assert result.name == "comprehensive"

    # Validate automation_spec (FixedRate)
    assert result.automation_spec.type == common_pb2.TriggerAutomationSpecType.TYPE_SCHEDULE
    assert result.automation_spec.schedule.rate.value == 30
    assert result.automation_spec.schedule.rate.unit == common_pb2.FixedRateUnit.FIXED_RATE_UNIT_MINUTE
    assert result.automation_spec.schedule.kickoff_time_input_arg == "trigger_time"

    # Validate trigger spec - active state
    assert result.spec.active is False

    # Validate run_spec - overwrite_cache
    assert result.spec.run_spec.overwrite_cache is True

    # Validate run_spec - interruptible
    assert result.spec.run_spec.interruptible is not None
    assert result.spec.run_spec.interruptible.value is True

    # Validate run_spec - queue
    assert result.spec.run_spec.cluster == "ml-queue"

    # Validate run_spec - labels
    assert result.spec.run_spec.labels is not None
    assert result.spec.run_spec.labels.values == {"component": "etl", "team": "data"}

    # Validate run_spec - annotations
    assert result.spec.run_spec.annotations is not None
    assert result.spec.run_spec.annotations.values == {"version": "v1.0"}

    # Validate run_spec - env_vars
    assert result.spec.run_spec.envs is not None
    env_dict = {kv.key: kv.value for kv in result.spec.run_spec.envs.values}
    assert env_dict == {"ENV": "staging"}

    # Validate inputs - should include trigger inputs that override task defaults
    # Expected: batch_size=100, mode="staging"
    # Task defaults that are NOT in trigger inputs: region="us-east", learning_rate=0.01
    assert len(result.spec.inputs.literals) == 4  # batch_size, mode, region, learning_rate

    # Create a dict of inputs for easy validation
    input_dict = {lit.name: lit.value for lit in result.spec.inputs.literals}

    # Trigger overrides
    assert "batch_size" in input_dict
    assert input_dict["batch_size"].scalar.primitive.integer == 100

    assert "mode" in input_dict
    assert input_dict["mode"].scalar.primitive.string_value == "staging"

    # Task defaults (not overridden by trigger)
    assert "region" in input_dict
    assert input_dict["region"].scalar.primitive.string_value == "us-east"

    assert "learning_rate" in input_dict
    assert input_dict["learning_rate"].scalar.primitive.float_value == 0.01
