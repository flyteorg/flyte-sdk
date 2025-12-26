import flyte

flyte.init_from_config()
from environment_picker import entrypoint

if __name__ == "__main__":
    r = flyte.run(entrypoint, n=5)
    print(r.url)
