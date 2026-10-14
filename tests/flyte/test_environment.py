from flyte._environment import Environment


def test_description_truncation():
    """Test that description longer than 255 characters is truncated."""
    long_description = "a" * 300

    env = Environment(name="test-env", description=long_description)

    # Verify that the description was truncated to 255 characters with ...(tr.) suffix
    assert len(env.description) == 255
    assert env.description == "a" * 247 + "...(tr.)"


def test_get_kwargs_copies_depends_on():
    """`_get_kwargs` must copy `depends_on`, not alias it, so a cloned/derived
    environment doesn't share (and mutate in place) the parent's dependency list."""
    dep = Environment(name="dep")
    env = Environment(name="parent", depends_on=[dep])

    kwargs = env._get_kwargs()

    # Distinct list object...
    assert kwargs["depends_on"] is not env.depends_on
    assert kwargs["depends_on"] == [dep]

    # ...so mutating the clone's list does not leak back to the parent.
    other = Environment(name="other")
    kwargs["depends_on"].append(other)
    assert other not in env.depends_on


def test_add_dependency_does_not_leak_through_get_kwargs():
    """Adding a dependency to a derived env (built from `_get_kwargs`) must not mutate
    the original env's `depends_on`."""
    original = Environment(name="orig")
    derived = Environment(**{**original._get_kwargs(), "name": "derived"})

    extra = Environment(name="extra")
    derived.add_dependency(extra)

    assert extra in derived.depends_on
    assert extra not in original.depends_on
