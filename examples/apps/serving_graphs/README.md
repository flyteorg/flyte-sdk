# Serving graphs

A *serving graph* is a set of Flyte apps that talk to each other inside the
cluster. Instead of stuffing one process with every stage of a request, you
split it into multiple `AppEnvironment`s that you deploy together — each one
sized for its own bottleneck (cheap CPU wide, expensive GPU narrow), with its
own image and its own scaling policy.

This directory has three examples that build up from the simplest two-app
chain to a realistic CPU/GPU inference split and an A/B-tested rollout.

## Core concepts

### 1. Deploying multiple apps together — `depends_on`

`depends_on=[other_env]` tells Flyte that `other_env` must be deployed
alongside this one. Calling `flyte.serve(root_env)` deploys the whole
dependency closure transitively, so you only ever name the entry-point app.

```python
gpu_env = FastAPIAppEnvironment(name="gpu-side",   ...)
cpu_env = FastAPIAppEnvironment(name="cpu-side",   ..., depends_on=[gpu_env])

flyte.serve(cpu_env)   # deploys both
```

`depends_on` is about deployment co-scheduling, not request-time ordering — at
runtime each app is independent.

### 2. Getting an upstream app's endpoint

Two patterns, both useful:

**Pattern A — `env.endpoint` (Python property).** When both apps live in the
same Python file (or module), the upstream env object is in scope, and you can
read `env.endpoint` directly. It resolves correctly in three contexts:

- locally → `http://localhost:<port>`
- inside the cluster → the private cluster-internal URL
- elsewhere → the public URL

```python
async with httpx.AsyncClient(base_url=gpu_env.endpoint) as client:
    await client.post("/infer", content=batch.tobytes())
```

This is what `image_classification.py` uses, since both apps are defined in
one file.

**Pattern B — `flyte.app.AppEndpoint` as a parameter.** When you'd rather not
import the upstream env (different file, different process, looking it up by
name), declare it as a `Parameter` and have Flyte inject the resolved URL via
an environment variable:

```python
env2 = FastAPIAppEnvironment(
    name="caller",
    ...,
    parameters=[
        flyte.app.Parameter(
            name="app1_url",
            value=flyte.app.AppEndpoint(app_name="callee"),
            env_var="APP1_URL",   # available as os.getenv("APP1_URL") at runtime
        ),
    ],
    depends_on=[env1],
)
```

`two_app_chain.py` demonstrates both patterns side by side.

### 3. Sizing each node independently

Each `AppEnvironment` carries its own image, resources, and scaling. That's
the entire point of splitting — the GPU side can stay narrow with
`scaling=Scaling(replicas=(1, 2))` while the CPU side scales wide with
`scaling=Scaling(replicas=(1, 8))`, with no shared autoscaling policy between
them.

## The three examples

### `two_app_chain.py` — the minimal chain

Two FastAPI apps where `app2` proxies HTTP calls to `app1`. Shows both
endpoint-discovery patterns (`env1.endpoint` and `AppEndpoint` parameter)
side by side, plus `depends_on` for co-deployment. Start here if you've never
built a serving graph before.

```bash
python two_app_chain.py
```

### `image_classification.py` — CPU pre/post + GPU forward

The canonical heterogeneous-resource pipeline:

```
client ──► [cpu_app  x N replicas]  ──► [gpu_app x M replicas] ──► back
            decode + resize + softmax     ResNet18 forward only
            cheap CPU, scale wide         expensive GPU, scale narrow
```

The two apps speak raw float32 bytes over `application/octet-stream` — for
anything tensor-shaped, skipping JSON is the single biggest perf knob.

```bash
python image_classification.py
# then:
curl -X POST $URL/classify -H 'content-type: application/json' \
  -d '{"image_url": "https://upload.wikimedia.org/wikipedia/commons/4/41/Sunflower_from_Silesia2.jpg"}'
```

### `ab_testing.py` — Statsig-driven variant routing

A root app uses [Statsig](https://www.statsig.com/) feature gates to route
each request to one of two variant apps (`app_a` / `app_b`), with consistent
per-user bucketing. Shows how a serving graph extends beyond load splitting
to traffic shaping for experiments.

The Statsig client lives in a singleton on the root app's lifespan; variant
apps are completely independent.

```bash
python ab_testing.py
# then visit `<endpoint>/` for the demo UI, or:
curl '<endpoint>/process/hello?user_key=user123'
```

**Setup before running:**

1. Get a Server Secret Key at [statsig.com](https://www.statsig.com/) → Settings → API Keys.
2. Create a feature gate named `variant_b` (e.g. 50% rollout).
3. Set the Flyte secret:
   ```bash
   flyte secrets set statsig-api-key STATSIG_API_KEY="your-secret-key-here"
   ```

The example reads the key via `flyte.Secret("statsig-api-key", as_env_var="STATSIG_API_KEY")`.

Response shape:

```json
{
  "ab_test_result": {
    "user_key": "user123",
    "selected_variant": "A",
    "gate_name": "variant_b"
  },
  "response": {
    "variant": "A",
    "message": "App A processed: hello",
    "algorithm": "fast-processing"
  }
}
```

Use stable identifiers (user ID, session ID) for `user_key` so the same user
always lands in the same bucket. To swap `check_gate` for an experiment or
dynamic config:

```python
experiment = statsig.get_experiment(user, "my_experiment")
variant = experiment.get("variant", "A")
```

## When to split into a serving graph

Split when stages have **different bottlenecks** (CPU vs GPU vs memory),
**different scaling needs** (bursty vs steady), **different lifecycles**
(model weights you don't want to reload), or **different routing concerns**
(A/B, canary). Don't split just to separate code — a single app with a few
endpoints is simpler to operate.
