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

from typing import Any, Optional


def sampling_params(
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    ignore_eos: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    so_args: Optional[dict[str, Any]] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a `SamplingParams`-compatible argument dict.

    Only the arguments you set are included, so anything left out falls back to
    DeepInfer's own `SamplingParams` defaults rather than a copy of them that could
    drift from upstream. Note that upstream's default `max_tokens` is small; set it
    explicitly for anything beyond short completions.

    Args:
        temperature: Sampling temperature (`0` for greedy).
        top_p: Nucleus sampling probability mass.
        top_k: Top-k cutoff (`-1` disables it upstream).
        max_tokens: Maximum tokens to generate per prompt.
        ignore_eos: Keep generating past the end-of-sequence token.
        top_logprobs: Number of top log-probabilities to return per token.
        so_args: Constrained-decoding config from `structured_output`.
        **extra: Any other `SamplingParams` fields, passed through untouched.
    """
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "top_logprobs": top_logprobs,
        "so_args": so_args,
    }
    return {k: v for k, v in params.items() if v is not None} | extra


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
