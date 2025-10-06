import flyte.git

flyte.init_from_config(flyte.git.config_from_root())
from environment_picker import entrypoint

if __name__ == "__main__":
    r = flyte.run(entrypoint, n=5)
    print(r.url)
