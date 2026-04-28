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


# --- include files -----------------------------------------------------------


def test_environment_include_default_is_empty_tuple():
    env = Environment(name="inc_default")
    assert env.include == ()
    assert isinstance(env.include, tuple)


def test_environment_include_accepts_tuple():
    inc = ("template.html", "config.yaml")
    env = Environment(name="inc_tuple", include=inc)
    assert env.include == inc
    assert isinstance(env.include, tuple)


def test_environment_include_normalizes_list_to_tuple():
    env = Environment(name="inc_list", include=["one.html", "two.html"])
    assert env.include == ("one.html", "two.html")
    assert isinstance(env.include, tuple)


def test_environment_include_preserves_order():
    env = Environment(name="inc_order", include=["c.txt", "a.txt", "b.txt"])
    assert env.include == ("c.txt", "a.txt", "b.txt")


def test_environment_include_accepts_generator():
    env = Environment(name="inc_gen", include=(p for p in ["x.html", "y.html"]))
    assert env.include == ("x.html", "y.html")


def test_environment_include_rejects_bare_string():
    # A bare string is a common foot-gun — Python would iterate characters — so
    # we reject it explicitly.
    with pytest.raises(TypeError, match="sequence of str paths"):
        Environment(name="inc_bare", include="only.html")


def test_environment_include_rejects_non_str_entries():
    with pytest.raises(TypeError, match="include entries must be str"):
        Environment(name="inc_badelem", include=["ok.html", 123])


def test_environment_include_populates_declaring_file():
    # When an environment is instantiated from a real user file, __post_init__
    # must record that file so include paths can be anchored later.
    env = Environment(name="inc_declaring", include=("a.html",))
    assert env._declaring_file is not None
    assert env._declaring_file.endswith("test_environment.py")


def test_environment_include_in_get_kwargs_when_nonempty():
    env = Environment(name="inc_kwargs", include=("t.html",))
    kwargs = env._get_kwargs()
    assert kwargs["include"] == ("t.html",)


def test_environment_include_omitted_from_get_kwargs_when_empty():
    env = Environment(name="inc_kwargs_empty")
    kwargs = env._get_kwargs()
    assert "include" not in kwargs


# --- hashability regression --------------------------------------------------
# These guard the invariant that the ``include`` field stays a hashable type
# (tuple). If someone ever regresses it back to ``list``, every assertion below
# will fail.


def test_environment_include_attr_is_hashable():
    env = Environment(name="inc_hash", include=("a.html", "b.html"))
    # If include regresses to list this raises TypeError.
    assert hash(env.include) == hash(("a.html", "b.html"))


def test_environment_include_equal_values_hash_equal():
    env_a = Environment(name="inc_hash_a", include=["a.html", "b.html"])
    env_b = Environment(name="inc_hash_b", include=("a.html", "b.html"))
    assert env_a.include == env_b.include
    assert hash(env_a.include) == hash(env_b.include)


def test_environment_include_usable_as_dict_key():
    env = Environment(name="inc_dictkey", include=("a.html",))
    lookup = {env.include: "ok"}
    assert lookup[("a.html",)] == "ok"


def test_environment_include_usable_in_set():
    env_a = Environment(name="inc_set_a", include=("a.html", "b.html"))
    env_b = Environment(name="inc_set_b", include=("a.html", "b.html"))
    env_c = Environment(name="inc_set_c", include=("other.html",))
    assert len({env_a.include, env_b.include, env_c.include}) == 2


def test_environment_empty_include_is_hashable():
    env = Environment(name="inc_empty_hash")
    assert hash(env.include) == hash(())
