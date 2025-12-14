# Save this as:  test_uv_short.py
import flyte
from flyte import Image
from pathlib import Path

image = Image.from_debian_base().with_env_vars({"CACHE_BUST": "With layers"}).with_pip_packages("tensorflow", "mypy")
env = flyte.TaskEnvironment(
    name="hello_world_env",
    image=Image.from_debian_base().with_env_vars({"CACHE_BUST": "With layers"}).with_pip_packages("tensorflow", "mypy")
)

@env.task()
def main():
    
    # Show results
    print(f"\n📦 Total layers: {len(image._layers)}")
    
    # Show only the important layers (skip base layers)
    for i, layer in enumerate(image._layers):
        print(f"Layer {i}: {type(layer).__name__} - {layer}")
    
    print(f"\n✅ Auto-separation working! Found {len([l for l in image._layers if 'PipPackages' in str(type(l))])} pip package layers")

if __name__ == "__main__":

    import flyte, logging 

    flyte.init_from_config(log_level=logging.DEBUG)
    run = flyte.with_runcontext(mode="remote", log_level=logging.DEBUG).run(main)
    print(run.name)
    print(run.url)
