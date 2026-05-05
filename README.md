> [!IMPORTANT]
> ## Flyte 2 Devbox is now available!
>
> Check out the guide [here](https://www.union.ai/docs/v2/flyte/user-guide/run-modes/running-devbox/) to get started.

---

# Flyte 2 SDK

**Reliably orchestrate ML pipelines, models, and agents at scale — in pure Python.**

[![Version](https://img.shields.io/pypi/v/flyte?label=version&color=blue)](https://pypi.org/project/flyte/)
[![Python](https://img.shields.io/pypi/pyversions/flyte?color=brightgreen)](https://pypi.org/project/flyte/)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange)](LICENSE)
[![Try in Browser](https://img.shields.io/badge/Try%20in%20Browser-Live%20Demo-7652a2)](https://flyte2intro.apps.demo.hosted.unionai.cloud/)
[![Docs](https://img.shields.io/badge/Docs-flyte-blue)](https://www.union.ai/docs/v2/flyte/user-guide/running-locally/)
[![SDK Reference](https://img.shields.io/badge/SDK%20Reference-API-brightgreen)](https://www.union.ai/docs/v2/union/api-reference/flyte-sdk/)
[![CLI Reference](https://img.shields.io/badge/CLI%20Reference-API-brightgreen)](https://www.union.ai/docs/v2/union/api-reference/flyte-cli/)

## Install

```bash
pip install flyte
```

## Example

```python
import asyncio
import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    image=flyte.Image.from_debian_base(python_version=(3, 12)),
)

@env.task
def calculate(x: int) -> int:
    return x * 2 + 5

@env.task
async def main(numbers: list[int]) -> float:
    results = await asyncio.gather(*[
        calculate.aio(num) for num in numbers
    ])
    return sum(results) / len(results)

if __name__ == "__main__":
    flyte.init()
    run = flyte.run(main, numbers=list(range(10)))
    print(f"Result: {run.result}")
```

<table>
<tr><td><b>Python</b></td><td><b>Flyte CLI</b></td></tr>
<tr>
<td>

```bash
python hello.py
```

</td>
<td>

```bash
flyte run hello.py main --numbers '[1,2,3]'
```

</td>
</tr>
</table>

## Serve a Model

```python
# serving.py
from fastapi import FastAPI
import flyte
from flyte.app.extras import FastAPIAppEnvironment

app = FastAPI()
env = FastAPIAppEnvironment(
    name="my-model",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi", "uvicorn"
    ),
)

@app.get("/predict")
async def predict(x: float) -> dict:
    return {"result": x * 2 + 5}

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(env)
```

<table>
<tr><td><b>Python</b></td><td><b>Flyte CLI</b></td></tr>
<tr>
<td>

```bash
python serving.py
```

</td>
<td>

```bash
flyte serve serving.py env
```

</td>
</tr>
</table>

### Local Development Experience

Install the TUI for a rich local development experience:

```bash
pip install flyte[tui]
```

[![Watch the local development experience](https://img.youtube.com/vi/lsfy-7DbbRM/maxresdefault.jpg)](https://www.youtube.com/watch?v=lsfy-7DbbRM)

## Learn More

- **[Live Demo](https://flyte2intro.apps.demo.hosted.unionai.cloud/)** — Try Flyte 2 in your browser
- **[Documentation](https://www.union.ai/docs/v2/flyte/user-guide/running-locally/)** — Get started running locally
- **[SDK Reference](https://www.union.ai/docs/v2/union/api-reference/flyte-sdk/)** — API reference docs
- **[CLI Reference](https://www.union.ai/docs/v2/union/api-reference/flyte-cli/)** — CLI docs
- **[Join the Flyte 2 Production Preview](https://www.union.ai/try-flyte-2)** — Get early access
- **[Features](FEATURES.md)** — Async parallelism, app serving, tracing, and more
- **[Examples](examples/)** — Ready-to-run examples for every feature
- **[Contributing](CONTRIBUTING.md)** — Set up a dev environment and contribute
- **[Slack](https://slack.flyte.org/)** | **[GitHub Discussions](https://github.com/flyteorg/flyte/discussions)** | **[Issues](https://github.com/flyteorg/flyte/issues)**

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Developing the Rust Core Controller

The Rust core controller (`flyte_controller_base`) ships as a separate PyPI wheel from the main
SDK. Keeping it separate avoids forcing the whole SDK build toolchain to become maturin-based;
we may revisit that decision once the Rust controller is the default execution path.

When iterating, keep `flyteidl2` pinned to the same version on both the Python and Rust sides
(see `pyproject.toml` and `rs_controller/Cargo.toml`).

### One-time builder setup

`cd` into `rs_controller` and run:

```bash
make build-builders
```

This builds the manylinux builder images once. They can be reused as the Rust source changes.

### Iteration cycle

Build the multi-arch wheels (linux/amd64 and linux/arm64):

```bash
cd rs_controller
make build-wheels
```

`make build-wheel-local` builds a macOS wheel for local Rust development.

`cd` back up to the repo root, then:

```bash
make dist
python maint_tools/build_default_image.py
```

Install the freshly-built wheel into your venv:

```bash
uv pip install --find-links ./rs_controller/dist --no-index --force-reinstall --no-deps flyte_controller_base
```

Repeat this loop — build new wheels, force-reinstall — to iterate.

### Build configuration

To support both Rust crate use and Python wheel distribution, the crate toggles the
`pyo3/extension-module` feature via Cargo features:

```toml
# rs_controller/Cargo.toml
[features]
default = ["pyo3/auto-initialize"]              # For local cargo run / cargo test (links libpython)
extension-module = ["pyo3/extension-module"]    # For Python wheels (no libpython linking)

[lib]
crate-type = ["rlib", "cdylib"]                 # Both Rust and Python consumers
```

`auto-initialize` embeds Python in Rust binaries — convenient on macOS where libpython is
linkable, but not available in manylinux. `extension-module` extends Python from Rust and
must *not* link libpython for wheel portability. The maturin pyproject opts into the
extension-module feature explicitly:

```toml
# rs_controller/pyproject.toml
[tool.maturin]
features = ["extension-module"]
```
