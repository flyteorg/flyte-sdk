from flyte._environment import Environment


def test_description_truncation():
    """Test that description longer than 255 characters is truncated."""
    long_description = "a" * 300

    env = Environment(name="test-env", description=long_description)

    # Verify that the description was truncated to 255 characters with ...(tr.) suffix
    assert len(env.description) == 255
    assert env.description == "a" * 247 + "...(tr.)"
