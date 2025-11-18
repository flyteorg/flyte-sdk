import pytest

import flyte


def test_trigger_description_truncation():
    """Test that trigger description is truncated to 255 characters"""
    # Create a description longer than 255 characters
    long_description = "A" * 300

    trigger = flyte.Trigger(
        name="test_truncation",
        automation=flyte.Cron("0 * * * *"),
        description=long_description,
    )

    # Description should be truncated to exactly 255 characters
    assert len(trigger.description) == 255
    assert trigger.description == "A" * 255

