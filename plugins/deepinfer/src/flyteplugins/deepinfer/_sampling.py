"""Sampling helpers for DeepInfer, without importing DeepInfer.

DeepInfer pins a specific `transformers`/`torch`/`tilelang` stack and must be
built from source, so the plugin deliberately does not depend on it at import time.
The builders in this module produce plain, serializable dictionaries that read
naturally and can be expanded into the upstream constructors at run time:

```python
from flyteplugins.deepinfer import sampling_params, structured_output

params = sampling_params(
    temperature=0,
    max_tokens=512,
    so_args=structured_output("json", json_schema_string),
)
outputs = llm.generate(prompts, sampling_params=[type(llm).SamplingParams(**params)])
```
"""

from __future__ import annotations

from typing import Any

# Defaults mirror `deepinfer.sampling_params.SamplingParams`.
_DEFAULTS: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "ignore_eos": False,
    "max_tokens": 16,
    "top_logprobs": 0,
}


def sampling_params(**kwargs: Any) -> dict[str, Any]:
    """Build a `SamplingParams`-compatible argument dict with DeepInfer defaults.

    Merge any keyword arguments over the upstream `SamplingParams` defaults so a task
    author never has to recall whether the framework's default `max_tokens` is 16,
    and can pass the result straight to `deepinfer.sampling_params.SamplingParams`.

    Args:
        **kwargs: Overrides; typical keys are `temperature`, `top_p`, `top_k`,
            `max_tokens`, `ignore_eos`, `top_logprobs` and `so_args` (from
            `structured_output`). Unknown keys are passed through untouched.
    """
    params = dict(_DEFAULTS)
    params.update(kwargs)
    return params


def structured_output(schema_type: str, schema: Any, **extra: Any) -> dict[str, Any]:
    """Build the `so_args` value for DeepInfer's constrained generation.

    DeepInfer constrains decoding to a regex, a JSON schema (serialized to a string),
    or an EBNF grammar by setting `so_args={"schema_type": ..., "schema": ...}` on a
    `SamplingParams`. This is a thin, import-free constructor for that dict and the
    optional trigger/ bound controls.

    Args:
        schema_type: One of `"regex"`, `"json"`, or `"ebnf"`.
        schema: The grammar; a **string** for `regex`/`ebnf`, and (per upstream) a
            JSON-serialized string for `json`.
        **extra: Optional `so_args` controls, e.g. `trigger_token_ids` or
            `bound_token_id`.

    Returns:
        `{"schema_type": ..., "schema": ..., **extra}`.
    """
    if schema_type not in ("regex", "json", "ebnf"):
        raise ValueError(f"schema_type must be 'regex', 'json' or 'ebnf', got {schema_type!r}")
    return {"schema_type": schema_type, "schema": schema, **extra}
