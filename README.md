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

Flyte 2 is licensed under the [Apache 2.0 License](LICENSE).

## Developing the Core Controller

Create a separate virtual environment for the Rust contoller inside the rs_controller folder. The reason for this is
because the rust controller should be a separate pypi package. The reason it should be a separate pypi package is that
including it into the main SDK as a core component means the entire build toolchain for the SDK will need to become
rust/maturin based. We should probably move to this model in the future though.

Keep important dependencies the same though, namely flyteidl2.

The following instructions are for helping to build the default multi-arch image. Each architecture needs a different wheel. Each wheel needs to be built by a different docker image.

### Setup Builders
`cd` into `rs_controller` and run `make build-builders`. This will build the builder images once, so you can keep using them as the rust code changes.

### Iteration Cycle
Run `make build-wheels` to actually build the multi-arch wheels. This command should probably be updated to build all three,
currently it only builds for linux/amd64 and linux/arm64... the `make build-wheel-local` command builds a macosx wheel,
unclear what the difference is between that and the arm64 one, and unclear if both are present, which one pip chooses.

`cd` back up to the root folder of this project and proceed with
```bash
make dist
python maint_tools/build_default_image.py
```

To install the wheel locally for testing, use the following command with your venv active.
```bash
uv pip install --find-links ./rs_controller/dist --no-index --force-reinstall --no-deps flyte_controller_base
```
Repeat this process to iterate - build new wheels, force reinstall the controller package.

### Build Configuration Summary

In order to support both Rust crate publication and Python wheel distribution, we have
to sometimes use and sometimes not use the 'pyo3/extension-module' feature. To do this, this
project's Cargo.toml itself can toggle this on and off.

  [features]
  default = ["pyo3/auto-initialize"]     # For Rust crate users (links to libpython)
  extension-module = ["pyo3/extension-module"]  # For Python wheels (no libpython linking)

The cargo file contains

  # Cargo.toml
  [lib]
  crate-type = ["rlib", "cdylib"]  # Support both Rust and Python usage

When using 'default', 'auto-initialize' is turned on, which requires linking to libpython, which exists on local Mac so
this works nicely. It is not available in manylinux however, so trying to build with this feature in a manylinux docker
image will fail. But that's okay, because the purpose of the manylinux container is to build wheels,
and for wheels, we need the 'extension-module' feature, which disables linking to libpython.

The key insight: auto-initialize is for embedding Python in Rust (needs libpython), while
extension-module is for extending Python with Rust (must NOT link libpython for portability).

This setup makes it possible to build wheels and also run Rust binaries with `cargo run --bin`. 

(not sure if this is needed)
  # pyproject.toml
  [tool.maturin]
  features = ["extension-module"]  # Tells maturin to use extension-module feature
  
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
