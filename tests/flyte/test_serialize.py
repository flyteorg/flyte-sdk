import flyte


def _env():
    env = flyte.TaskEnvironment(name="serialize_test")

    @env.task
    def greet(i: int) -> str:
        return f"hello union {i}"

    @env.task
    def main(n: int = 3) -> list[str]:
        return [greet(i) for i in range(n)]

    return env, main, greet


def test_serialize_single_task_is_code_agnostic():
    env, main, greet = _env()
    spec = flyte.serialize(main)
    assert spec.task_template.id.name  # a name is set
    args = list(spec.task_template.container.args)
    # No code bundle is baked in: those args are injected per-tenant later.
    assert "--tgz" not in args
    assert "--pkl" not in args
    # Container has an image URI resolved offline.
    assert spec.task_template.container.image


def test_serialize_env_returns_every_task():
    env, main, greet = _env()
    specs = flyte.serialize_env(env)
    names = {s.task_template.id.name for s in specs}
    # Extract the bare function names (after the last dot) from fully-qualified names
    bare = {n.split(".")[-1] for n in names}
    assert {"main", "greet"} <= bare


def test_serialize_default_inputs_captured():
    env, main, greet = _env()
    spec = flyte.serialize(main)
    # main(n: int = 3) -> the default is captured so CreateRun can launch with no inputs.
    assert len(spec.default_inputs) >= 1
    # Verify the captured default corresponds to the 'n' parameter with value 3
    param_names = {param.name for param in spec.default_inputs}
    assert "n" in param_names
    # Find the 'n' parameter and verify its literal value is 3
    n_param = next((p for p in spec.default_inputs if p.name == "n"), None)
    assert n_param is not None
    assert n_param.parameter.default.scalar.primitive.integer == 3
