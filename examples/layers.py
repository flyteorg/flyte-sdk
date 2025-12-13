from flyte import Image
import flyte

image = flyte.Image.from_debian_base().with_pip_packages("tensorflow", "mypy")

# Use it in a task environment
env = flyte.TaskEnvironment(name="test-env", image=image)

@env.task
def simple_task() -> str:
    print(f"Layers created: {len(image._layers)}")
    for i, layer in enumerate(image._layers):
        print(f"Layer {i+1}: {type(layer).__name__} - {layer}")
    return "OK"


if __name__ == "__main__":

    import flyte
    import logging

    flyte.init(
        endpoint="dns:///tryv2.hosted.unionai.cloud",
        insecure=False,
        auth_type="Pkce",
        org="tryv2",
        project="taiwan",
        domain="development",
        image_builder="remote",
    )
    run = flyte.with_runcontext(mode="remote", log_level=logging.DEBUG).run(simple_task)
    print(run.name)
    print(run.url)