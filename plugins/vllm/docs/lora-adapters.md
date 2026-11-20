# Plan: LoRA adapters on `VLLMAppEnvironment`

Status: plan, nothing implemented. Base branch: `main`.

## Goal

Serve one base model and many LoRA adapters from a single vLLM app, with two ways to get an adapter in:

1. **Static**: adapters known at deploy time, loaded at startup.
2. **Dynamic**: an adapter that appears in object storage after the app is deployed is served on
   first request, with no redeploy and no upload/load step.

The motivating workflow is evaluation of externally trained adapters. A training task writes a PEFT
directory (`adapter_config.json` + `adapter_model.safetensors`), and an eval task calls the app with
`model="<adapter-name>"`. The base model and adapter have to live in the same GPU memory, so the
adapter must be loaded server-side; this is what hosted providers do with an "upload adapter, then
load it onto a deployment" flow.

## What vLLM provides

Checked against the current vLLM LoRA docs (`docs.vllm.ai/en/latest/features/lora.html`), not yet
against the pinned `VLLM_MIN_VERSION` (0.26.0). See [Validation](#validation-before-implementation).

| Mechanism | How | Notes |
|---|---|---|
| Startup adapters | `--enable-lora --lora-modules <name>=<path>` or a JSON object with `name`, `path`, `base_model_name` | Each adapter is a `model` name on the OpenAI endpoint |
| Runtime API | `POST /v1/load_lora_adapter` `{lora_name, lora_path}`, `POST /v1/unload_lora_adapter` | Needs `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. `load_inplace: true` replaces an adapter of the same name |
| Resolver plugin | `LoRAResolver.resolve_lora(base_model_name, lora_name) -> LoRARequest`, registered with `LoRAResolverRegistry.register_resolver(name, resolver)` | Called when a request names an unknown model. Also needs `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` |
| Capacity | `--max-loras`, `--max-cpu-loras`, `--max-lora-rank` | Fixed at startup. Docs recommend `max-lora-rank` equal to the largest adapter rank, not larger |

vLLM ships two resolvers (`lora_filesystem_resolver`, `lora_hf_hub_resolver`). Neither reads object
storage, and the docs call the HF one insecure for production.

## Proposed API

```python
from flyteplugins.vllm import LoRAAdapter, VLLMAppEnvironment

VLLMAppEnvironment(
    name="qwen-lora",
    model_path=base_model,          # unchanged
    model_id="qwen3-8b",
    # static: resolved at deploy time, downloaded, passed as --lora-modules
    lora_adapters=(
        LoRAAdapter(name="sql-v1", path=RunOutput(...)),
        LoRAAdapter(name="support", path="s3://bucket/loras/support"),
        LoRAAdapter(name="hub", hf_path="org/some-adapter"),
    ),
    # dynamic: model="<name>" resolves to <lora_prefix>/<name>/ on first request
    lora_prefix="s3://bucket/loras/",
    max_loras=4,
    max_lora_rank=64,
    max_cpu_loras=None,
)
```

- `LoRAAdapter` is a frozen dataclass with `name` and exactly one of `path`
  (`str | RunOutput | ArtifactValue`) or `hf_path`. `lora_adapters` is a tuple so the environment
  stays hashable.
- Setting either `lora_adapters` or `lora_prefix` adds `--enable-lora`. The `max_*` fields map to the
  matching flags and are rejected when LoRA is not enabled.
- Passing any of these flags through `extra_args` as well is an error, same as `args` today.
- `clone_with` gains the new fields.

## Design

### Static adapters

Each adapter with a `path` becomes a `Parameter(name="lora_<name>", download=True,
mount="/tmp/flyte/loras/<name>")`, the same shape `draft_model_path` uses now. An `ArtifactValue`
therefore records the app → adapter lineage edge. Each adapter is rendered as one JSON
`--lora-modules` value with `base_model_name=model_id`; the JSON already survives `fserve`'s shell
join through `_shell_safe`. `hf_path` adapters pass the repo id as the path and vLLM downloads them.

### Dynamic adapters: `FlyteLoRAResolver`

New module `flyteplugins/vllm/_lora_resolver.py`, registered through a second
`vllm.general_plugins` entry point. It is a no-op unless `FLYTE_LORA_REMOTE_PREFIX` is set, which
`VLLMAppEnvironment` does when `lora_prefix` is given (along with
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`).

`resolve_lora(base_model_name, lora_name)`:

1. Validate `lora_name` against `^[A-Za-z0-9][A-Za-z0-9._-]*$`. One path segment, so a request
   cannot escape the prefix. Anything else returns `None` (model not found).
2. Take a per-name `asyncio.Lock` so concurrent first requests download once.
3. If `<cache>/<name>/adapter_config.json` is missing, download `<prefix>/<name>/` with
   `flyte.storage` into a temp directory and rename it into place. A missing remote directory
   returns `None`.
4. Return `LoRARequest(lora_name, lora_path, lora_int_id)` with an id derived from a stable hash of
   the name, not Python's seeded `hash()`.

Heavy imports (`vllm`, `flyte.storage`) stay function-local.

### Why the resolver and not the runtime API

A `load_lora_adapter` call reaches one replica, and what it loaded is gone after scale-to-zero or a
restart. The resolver runs in every replica and reloads on demand, so autoscaling needs no
coordination. The caller also never supplies a path, only a name under a prefix the deployer chose.

### Security

Resolvers only work with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, and that flag also turns on the
load/unload endpoints, which take an arbitrary `lora_path` from the caller. vLLM's docs say to use
this only in a trusted environment. Plan:

- `lora_prefix` with `requires_auth=False` raises at construction.
- Investigate blocking `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` with a small ASGI
  middleware passed via `--middleware`, so the resolver is the only way in. If that works it is on
  by default.

### Behaviour to document

- Adapters are cached by name, on disk and inside vLLM. Overwriting `<prefix>/<name>/` is not picked
  up by running replicas. Use immutable names (`<run>-<step>`).
- The first request for an adapter pays the download and load. Adapters are usually tens to hundreds
  of MB.
- `max_lora_rank` is fixed at startup. An adapter with a larger rank fails to load.
- The local cache is not evicted in the first version; it is bounded by the pod's `disk=`.

## Validation before implementation

Run on our demo tenant with a tiny base model and a tiny adapter, against vLLM 0.26.0.

1. `LoRAResolver` / `LoRARequest` signatures and the registration call match 0.26.0.
2. A `vllm.general_plugins` entry point runs in the API server process, where resolvers are consulted.
3. LoRA works with `--load-format flyte-vllm-streaming`. Adapter loading should be a separate code
   path from the base model loader, but the draft-model case already showed the streaming loader is
   process-wide, so this needs a real run. Fallback: `stream_model=False` when LoRA is enabled, with
   a warning.
4. LoRA works on a sharded (tensor-parallel) base model loaded through `_load_sharded_model`.
5. `--middleware` can reject the load/unload routes without breaking the resolver.
6. Resolver returning `None` produces a clean 404 for unknown names.
7. Latency of a first request to a new adapter, and of switching between `max_loras + 1` adapters.

## Phases

1. **Spike**: items 1 to 6 above with hand-written flags and env vars on the existing
   `VLLMAppEnvironment` (`extra_args`, `env_vars`). No SDK change. Results go into this doc.
2. **Static adapters**: `LoRAAdapter`, `lora_adapters`, `max_*`, `clone_with`, unit tests for args
   and parameters, README section.
3. **Dynamic adapters**: `FlyteLoRAResolver`, `lora_prefix`, endpoint blocking, unit tests with a
   fake store.
4. **Example**: `examples/genai/vllm/` train-tiny-adapter → eval-against-app, run end to end on the
   demo tenant. Tutorial in `unionai-examples` afterwards.

Phases 2 and 3 are separate PRs.

## Not in this plan

- Resolving adapters by artifact name through the v2 ArtifactService (lineage, base-model
  compatibility check). Natural follow-up once the prefix resolver exists.
- In-place refresh of a named adapter (`load_inplace`) for RL loops where the adapter changes every
  few steps.
- Cache eviction, per-adapter metrics, routing requests to replicas that already hold an adapter.
- LoRA for `flyteplugins-sglang` and `flyteplugins-llamacpp`.

## Related PRs

- #1562 (llama.cpp delivery modes) and its stack: independent. It removed the shared
  `ModelDeliveryMixin`, so there is no common delivery surface to build on.
- #1623 (vLLM + LMCache example): independent. Adapters change the KV cache key space, so the two
  should be tested together before recommending both.
