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

## Rust Controller (experimental)

The Rust controller is an alternative implementation of the remote controller written in Rust and exposed
to Python via maturin / pyo3. Distributed as a separate `flyte_controller_base` wheel so the main SDK does
not need to switch its build toolchain to rust/maturin. Keep important dependencies (notably `flyteidl2`)
in lockstep between `pyproject.toml`, `rs_controller/pyproject.toml`, and `rs_controller/Cargo.toml`.

### Running with the Rust controller

The Rust controller is gated behind an env var. Set it to `1` (also accepts `true` / `yes`):

```bash
_F_USE_RUST_CONTROLLER=1 python examples/basics/hello_v2.py
```

The driver propagates this env var to all sub-task pods, so both the driver and child actions use the
Rust controller for that run.

> **v1 limitations.** The Rust controller currently supports only the legacy
> QueueService + StateService path. Do **not** combine `_F_USE_RUST_CONTROLLER=1` with
> `_U_USE_ACTIONS=1` until ActionsService support lands. Other gaps tracked as follow-ups:
> abort RPC on cancel, trace-action enqueue, `Code.ABORTED` fast-fail, tunable retries / QPS,
> graceful `stop()`. See PR #675.

> Dev iteration requires the local image builder. The `flyte_controller_base` wheel is not
> on PyPI until release, and the remote image builder installs all wheels in a layer at once,
> so it cannot resolve `flyte_controller_base` from a sibling layer. Use the local image
> builder while developing the Rust controller:
>
> ```yaml
> # .flyte/config.yaml
> image:
>   builder: local
> ```

### Developing the Rust controller

#### One-time setup

Build the manylinux builder images. They are cached, so you only need to rebuild them when the
build tooling itself changes:

```bash
cd rs_controller
make build-builders
cd ..
```

#### Iteration loop

After every Rust change, run the all-in-one dev target from the repo root:

```bash
REGISTRY=<your-registry> make dev-rs-dist
```

`dev-rs-dist` does four things:

1. `cd rs_controller && make build-wheels` — build manylinux x86_64 + aarch64 wheels (use
   `make build-wheel-local` if you only need a macOS wheel for the driver).
2. `make dist` — build the main `flyte` SDK wheel.
3. `uv run python maint_tools/build_default_image.py --registry $(REGISTRY)` — build the default
   image with both wheels baked in and push it to your registry.
4. `uv pip install --find-links ./rs_controller/dist --no-index --force-reinstall --no-deps flyte_controller_base` —
   refresh the wheel in your local venv so the driver picks up the new build.

After this, any `flyte.TaskEnvironment` that does not pass an explicit `image=` will resolve to the default
debian image and automatically have the Rust wheel layered in. If you do pass an explicit `image=`, the
auto-bake is skipped; in that case, chain `.with_local_rs_controller()` onto the image to bake the Rust wheel
manually.

If you only changed Python (not Rust), you can skip the wheel rebuild and just run `make dist` plus
the rebuild image step. The Rust wheel is reused.

### Build configuration summary

The Rust crate ships with two cargo features so the same project can produce a Rust rlib and a
Python extension wheel:

```toml
[features]
default = ["pyo3/auto-initialize"]            # Rust crate users; links libpython
extension-module = ["pyo3/extension-module"]  # Python wheels; no libpython linking

[lib]
crate-type = ["rlib", "cdylib"]               # Both Rust and Python usage
```

- `pyo3/auto-initialize` embeds Python into Rust (works locally on macOS, fails inside the manylinux
  builder because libpython is unavailable there).
- `pyo3/extension-module` extends Python with Rust (must not link libpython for portable wheels).

So local `cargo run --bin <name>` uses `default` features, and the manylinux builder explicitly
disables defaults and turns on `extension-module`:

```toml
# rs_controller/pyproject.toml
[tool.maturin]
no-default-features = true
features = ["extension-module"]
```

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
