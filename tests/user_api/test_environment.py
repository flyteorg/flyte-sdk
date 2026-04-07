import pytest

import flyte
from flyte._environment import Environment, list_loaded_environments


def test_environment_basic():
    env = Environment(name="basic_env")
    assert env.name == "basic_env"
    assert env.image == "auto"
    assert env.resources is None
    assert env.depends_on == []
    assert env.interruptible is False


def test_environment_with_resources():
    res = flyte.Resources(cpu="2", memory="1Gi")
    env = Environment(name="with_resources", resources=res)
    assert env.resources == res


def test_environment_with_env_vars():
    env = Environment(name="with_env_vars", env_vars={"KEY": "VALUE"})
    assert env.env_vars == {"KEY": "VALUE"}


def test_environment_invalid_name():
    with pytest.raises(ValueError, match="must be in snake_case or kebab-case format"):
        Environment(name="Invalid Name!")


def test_environment_valid_snake_case():
    env = Environment(name="valid_name")
    assert env.name == "valid_name"


def test_environment_valid_kebab_case():
    env = Environment(name="valid-name")
    assert env.name == "valid-name"


def test_environment_valid_with_numbers():
    env = Environment(name="env123")
    assert env.name == "env123"


def test_environment_invalid_image_type():
    with pytest.raises(TypeError, match="Expected image to be of type str or Image"):
        Environment(name="bad_image", image=123)


def test_environment_invalid_resources_type():
    with pytest.raises(TypeError, match="Expected resources to be of type Resources"):
        Environment(name="bad_resources", resources=123)


def test_environment_invalid_env_vars_type():
    with pytest.raises(TypeError, match="Expected env_vars to be of type Dict"):
        Environment(name="bad_env_vars", env_vars=123)


def test_environment_invalid_depends_on_type():
    with pytest.raises(TypeError, match="Expected depends_on to be of type List"):
        Environment(name="bad_depends", depends_on=[123])


def test_environment_add_dependency():
    env1 = Environment(name="env1")
    env2 = Environment(name="env2")
    env1.add_dependency(env2)
    assert env2 in env1.depends_on


def test_environment_add_self_dependency():
    env = Environment(name="self_dep")
    with pytest.raises(ValueError, match="Cannot add self as a dependency"):
        env.add_dependency(env)


def test_environment_add_invalid_dependency():
    env = Environment(name="env_dep")
    with pytest.raises(TypeError, match="Expected Environment"):
        env.add_dependency(123)


def test_environment_description_truncation():
    long_desc = "x" * 300
    env = Environment(name="long_desc", description=long_desc)
    assert len(env.description) <= 255


def test_environment_registered():
    registry_before = len(list_loaded_environments())
    Environment(name="registered_env")
    assert len(list_loaded_environments()) == registry_before + 1


def test_environment_get_kwargs():
    env = Environment(
        name="kwargs_test",
        resources=flyte.Resources(cpu="1"),
        env_vars={"A": "B"},
        secrets="my-secret",
    )
    kwargs = env._get_kwargs()
    assert "resources" in kwargs
    assert "env_vars" in kwargs
    assert "secrets" in kwargs
    assert "depends_on" in kwargs
    assert "image" in kwargs


def test_environment_interruptible():
    env = Environment(name="interruptible_env", interruptible=True)
    assert env.interruptible is True


def test_config_fingerprint_deterministic():
    env1 = Environment(name="fingerprint_env", env_vars={"A": "1"})
    env2 = Environment(name="fingerprint_env", env_vars={"A": "1"})
    assert env1.config_fingerprint() == env2.config_fingerprint()


def test_config_fingerprint_changes_with_config():
    env1 = Environment(name="fp_env1", env_vars={"A": "1"})
    env2 = Environment(name="fp_env1", env_vars={"A": "2"})
    assert env1.config_fingerprint() != env2.config_fingerprint()


def test_config_fingerprint_includes_all_init_fields():
    env = Environment(
        name="fp_full",
        image="python:3.12",
        resources=flyte.Resources(cpu="2"),
        env_vars={"KEY": "VAL"},
        interruptible=True,
    )
    fp = env.config_fingerprint()
    assert "name=" in fp
    assert "image=" in fp
    assert "resources=" in fp
    assert "env_vars=" in fp
    assert "interruptible=" in fp


def test_config_fingerprint_skips_objects_with_memory_addresses():
    """Objects whose repr contains 0x (memory addresses) are excluded."""
    env = Environment(name="fp_addr")
    fp = env.config_fingerprint()
    assert "0x" not in fp
