> [!IMPORTANT]
> ## Flyte 2 Devbox is now available!
>
> Check out the guide [here](https://www.union.ai/docs/v2/flyte/user-guide/run-modes/running-devbox/) to get started.
>
> Looking for Flyte 1? Go to the [master](https://github.com/flyteorg/flyte/tree/master) branch, where Flyte 1 is now maintained.

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
