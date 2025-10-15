from datetime import datetime

import pytest

import flyte


def test_trigger_hourly():
    """Test Trigger.hourly() convenience method"""
    trigger = flyte.Trigger.hourly()

    assert trigger.name == "hourly"
    assert isinstance(trigger.automation, flyte.Cron)
    assert trigger.automation.expression == "0 * * * *"
    assert trigger.inputs == {"trigger_time": flyte.TriggerTime}
    assert trigger.auto_activate is True


def test_trigger_hourly_custom():
    """Test Trigger.hourly() with custom parameters"""
    trigger = flyte.Trigger.hourly(
        trigger_time_input_key="scheduled_at",
        name="custom_hourly",
        description="Custom hourly trigger",
        auto_activate=False,
        inputs={"batch_size": 100},
        env_vars={"ENV": "prod"},
        interruptible=True,
        overwrite_cache=True,
        queue="high-priority",
        labels={"team": "ml"},
        annotations={"owner": "data-team"},
    )

    assert trigger.name == "custom_hourly"
    assert trigger.description == "Custom hourly trigger"
    assert trigger.auto_activate is False
    assert trigger.inputs == {"scheduled_at": flyte.TriggerTime, "batch_size": 100}
    assert trigger.env_vars == {"ENV": "prod"}
    assert trigger.interruptible is True
    assert trigger.overwrite_cache is True
    assert trigger.queue == "high-priority"
    assert trigger.labels == {"team": "ml"}
    assert trigger.annotations == {"owner": "data-team"}


def test_trigger_daily():
    """Test Trigger.daily() convenience method"""
    trigger = flyte.Trigger.daily()

    assert trigger.name == "daily"
    assert isinstance(trigger.automation, flyte.Cron)
    assert trigger.automation.expression == "0 0 * * *"
    assert trigger.inputs == {"trigger_time": flyte.TriggerTime}


def test_trigger_minutely():
    """Test Trigger.minutely() convenience method"""
    trigger = flyte.Trigger.minutely()

    assert trigger.name == "minutely"
    assert isinstance(trigger.automation, flyte.Cron)
    assert trigger.automation.expression == "* * * * *"
    assert trigger.inputs == {"trigger_time": flyte.TriggerTime}


def test_trigger_weekly():
    """Test Trigger.weekly() convenience method"""
    trigger = flyte.Trigger.weekly()

    assert trigger.name == "weekly"
    assert isinstance(trigger.automation, flyte.Cron)
    assert trigger.automation.expression == "0 0 * * 0"
    assert trigger.inputs == {"trigger_time": flyte.TriggerTime}


def test_trigger_monthly():
    """Test Trigger.monthly() convenience method"""
    trigger = flyte.Trigger.monthly()

    assert trigger.name == "monthly"
    assert isinstance(trigger.automation, flyte.Cron)
    assert trigger.automation.expression == "0 0 1 * *"
    assert trigger.inputs == {"trigger_time": flyte.TriggerTime}


def test_trigger_custom_cron():
    """Test custom Cron trigger"""
    trigger = flyte.Trigger(
        name="every_15_minutes",
        automation=flyte.Cron("*/15 * * * *"),
        description="Runs every 15 minutes",
    )

    assert trigger.name == "every_15_minutes"
    assert trigger.automation.expression == "*/15 * * * *"
    assert trigger.description == "Runs every 15 minutes"


def test_trigger_fixed_rate():
    """Test FixedRate trigger"""
    trigger = flyte.Trigger(
        name="every_30_minutes",
        automation=flyte.FixedRate(interval_minutes=30),
    )

    assert trigger.name == "every_30_minutes"
    assert isinstance(trigger.automation, flyte.FixedRate)
    assert trigger.automation.interval_minutes == 30
    assert trigger.automation.start_time is None


def test_trigger_fixed_rate_with_start_time():
    """Test FixedRate trigger with start time"""
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    trigger = flyte.Trigger(
        name="scheduled_rate",
        automation=flyte.FixedRate(interval_minutes=60, start_time=start_time),
    )

    assert trigger.automation.interval_minutes == 60
    assert trigger.automation.start_time == start_time


def test_trigger_with_inputs():
    """Test trigger with custom inputs"""
    trigger = flyte.Trigger(
        name="with_inputs",
        automation=flyte.Cron("0 * * * *"),
        inputs={
            "trigger_time": flyte.TriggerTime,
            "batch_size": 100,
            "mode": "production",
        },
    )

    assert trigger.inputs["trigger_time"] is flyte.TriggerTime
    assert trigger.inputs["batch_size"] == 100
    assert trigger.inputs["mode"] == "production"


def test_trigger_validation_empty_name():
    """Test that trigger name cannot be empty"""
    with pytest.raises(ValueError, match="Trigger name cannot be empty"):
        flyte.Trigger(name="", automation=flyte.Cron("0 * * * *"))


def test_trigger_validation_none_automation():
    """Test that automation cannot be None"""
    with pytest.raises(ValueError, match="Automation cannot be None"):
        flyte.Trigger(name="test", automation=None)


def test_task_with_single_trigger():
    """Test task decorated with a single trigger"""
    env = flyte.TaskEnvironment(name="test_env")

    trigger = flyte.Trigger.hourly()

    @env.task(triggers=trigger)
    async def task_with_trigger(trigger_time: datetime, x: int = 1) -> str:
        return f"Executed at {trigger_time.isoformat()}"

    assert task_with_trigger.triggers == (trigger,)
    assert len(task_with_trigger.triggers) == 1


def test_task_with_multiple_triggers():
    """Test task decorated with multiple triggers"""
    env = flyte.TaskEnvironment(name="test_env")

    trigger1 = flyte.Trigger.hourly()
    trigger2 = flyte.Trigger.daily()

    @env.task(triggers=(trigger1, trigger2))
    async def task_with_triggers(trigger_time: datetime, x: int = 1) -> str:
        return f"Executed at {trigger_time.isoformat()}"

    assert task_with_triggers.triggers == (trigger1, trigger2)
    assert len(task_with_triggers.triggers) == 2


def test_task_with_trigger_defaults_no_overlap():
    """Test task with defaults that don't overlap with trigger inputs"""
    env = flyte.TaskEnvironment(name="test_env")

    # Task has defaults for x and y
    # Trigger provides trigger_time (TriggerTime) and batch_size
    trigger = flyte.Trigger(
        name="custom",
        automation=flyte.Cron("0 * * * *"),
        inputs={
            "trigger_time": flyte.TriggerTime,
            "batch_size": 50,
        },
    )

    @env.task(triggers=trigger)
    async def task_no_overlap(trigger_time: datetime, batch_size: int, x: int = 10, y: str = "default") -> str:
        return f"{trigger_time}, {batch_size}, {x}, {y}"

    # Task should have the trigger
    assert task_no_overlap.triggers == (trigger,)

    # Trigger inputs should specify batch_size=50
    assert trigger.inputs["batch_size"] == 50
    assert trigger.inputs["trigger_time"] is flyte.TriggerTime


def test_task_with_trigger_defaults_partial_overlap():
    """Test task with defaults that partially overlap with trigger inputs"""
    env = flyte.TaskEnvironment(name="test_env")

    # Task has defaults: x=10, y="default", z=5
    # Trigger overrides: x=20, adds batch_size=100
    # y and z should use task defaults
    trigger = flyte.Trigger(
        name="partial_overlap",
        automation=flyte.Cron("0 0 * * *"),
        inputs={
            "trigger_time": flyte.TriggerTime,
            "x": 20,  # Override task default
            "batch_size": 100,  # New input not in task defaults
        },
    )

    @env.task(triggers=trigger)
    async def task_partial_overlap(
        trigger_time: datetime,
        x: int = 10,
        y: str = "default",
        z: int = 5,
        batch_size: int = 1,
    ) -> str:
        return f"{trigger_time}, {x}, {y}, {z}, {batch_size}"

    assert task_partial_overlap.triggers == (trigger,)
    assert trigger.inputs["x"] == 20  # Override
    assert trigger.inputs["batch_size"] == 100  # New
    # y and z not in trigger inputs, will use task defaults


def test_task_with_trigger_defaults_full_overlap():
    """Test task with defaults that fully overlap with trigger inputs"""
    env = flyte.TaskEnvironment(name="test_env")

    # Task has defaults: x=10, y=20
    # Trigger overrides both
    trigger = flyte.Trigger(
        name="full_overlap",
        automation=flyte.Cron("0 0 * * *"),
        inputs={
            "trigger_time": flyte.TriggerTime,
            "x": 100,
            "y": 200,
        },
    )

    @env.task(triggers=trigger)
    async def task_full_overlap(trigger_time: datetime, x: int = 10, y: int = 20) -> str:
        return f"{trigger_time}, {x}, {y}"

    assert task_full_overlap.triggers == (trigger,)
    assert trigger.inputs["x"] == 100
    assert trigger.inputs["y"] == 200


def test_task_with_multiple_triggers_different_defaults():
    """Test task with multiple triggers that provide different default values"""
    env = flyte.TaskEnvironment(name="test_env")

    # First trigger: hourly with batch_size=50
    trigger1 = flyte.Trigger(
        name="hourly_small_batch",
        automation=flyte.Cron("0 * * * *"),
        inputs={
            "trigger_time": flyte.TriggerTime,
            "batch_size": 50,
        },
    )

    # Second trigger: daily with batch_size=1000
    trigger2 = flyte.Trigger(
        name="daily_large_batch",
        automation=flyte.Cron("0 0 * * *"),
        inputs={
            "scheduled_at": flyte.TriggerTime,
            "batch_size": 1000,
        },
    )

    @env.task(triggers=(trigger1, trigger2))
    async def task_multi_triggers(trigger_time: datetime, scheduled_at: datetime, batch_size: int = 10) -> str:
        return f"{trigger_time}, {scheduled_at}, {batch_size}"

    assert len(task_multi_triggers.triggers) == 2
    assert task_multi_triggers.triggers[0].inputs["batch_size"] == 50
    assert task_multi_triggers.triggers[1].inputs["batch_size"] == 1000


def test_task_with_trigger_no_inputs():
    """Test task with trigger that has no custom inputs, only TriggerTime"""
    env = flyte.TaskEnvironment(name="test_env")

    trigger = flyte.Trigger.daily()

    @env.task(triggers=trigger)
    async def task_simple(trigger_time: datetime, x: int = 1, y: str = "hello") -> str:
        return f"{trigger_time}, {x}, {y}"

    # Trigger should only have TriggerTime, task defaults remain
    assert trigger.inputs == {"trigger_time": flyte.TriggerTime}


def test_task_with_trigger_env_vars():
    """Test task with trigger that specifies environment variables"""
    env = flyte.TaskEnvironment(name="test_env")

    trigger = flyte.Trigger(
        name="with_env",
        automation=flyte.Cron("0 * * * *"),
        inputs={"trigger_time": flyte.TriggerTime},
        env_vars={
            "DATABASE_URL": "postgres://prod",
            "LOG_LEVEL": "INFO",
        },
    )

    @env.task(triggers=trigger)
    async def task_with_env(trigger_time: datetime) -> str:
        return "ok"

    assert trigger.env_vars["DATABASE_URL"] == "postgres://prod"
    assert trigger.env_vars["LOG_LEVEL"] == "INFO"


def test_task_with_trigger_all_options():
    """Test task with trigger that uses all available options"""
    env = flyte.TaskEnvironment(name="test_env")

    trigger = flyte.Trigger(
        name="comprehensive",
        automation=flyte.FixedRate(interval_minutes=30),
        description="A comprehensive trigger",
        auto_activate=False,
        inputs={
            "trigger_time": flyte.TriggerTime,
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
    ) -> str:
        return "ok"

    assert trigger.name == "comprehensive"
    assert trigger.description == "A comprehensive trigger"
    assert trigger.auto_activate is False
    assert trigger.inputs["batch_size"] == 100
    assert trigger.inputs["mode"] == "staging"
    assert trigger.env_vars == {"ENV": "staging"}
    assert trigger.interruptible is True
    assert trigger.overwrite_cache is True
    assert trigger.queue == "ml-queue"
    assert trigger.labels == {"component": "etl", "team": "data"}
    assert trigger.annotations == {"version": "v1.0"}


def test_task_no_triggers():
    """Test task without any triggers"""
    env = flyte.TaskEnvironment(name="test_env")

    @env.task
    async def task_no_trigger(x: int, y: int) -> int:
        return x + y

    assert task_no_trigger.triggers == ()


def test_trigger_different_key_names():
    """Test triggers with different TriggerTime key names"""
    env = flyte.TaskEnvironment(name="test_env")

    trigger1 = flyte.Trigger.hourly(trigger_time_input_key="scheduled_at")
    trigger2 = flyte.Trigger.daily(trigger_time_input_key="execution_time")

    @env.task(triggers=(trigger1, trigger2))
    async def task_diff_keys(scheduled_at: datetime, execution_time: datetime, x: int = 1) -> str:
        return "ok"

    assert trigger1.inputs["scheduled_at"] is flyte.TriggerTime
    assert trigger2.inputs["execution_time"] is flyte.TriggerTime
    assert "trigger_time" not in trigger1.inputs
    assert "trigger_time" not in trigger2.inputs


@pytest.mark.asyncio
async def test_trigger_remote():
    import flyte.remote

    flyte.init_from_config("/Users/ytong/go/src/github.com/flyteorg/flyte-sdk/.flyte/playground.yaml")

    result = flyte.remote.Trigger.listall.aio(task_name="base.test_some_stuff")
    try:
        async for r in result:
            print(r)
    except Exception as e:
        print(f"Error fetching triggers: {e}")