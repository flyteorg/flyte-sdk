"""Stand-in for `deepinfer.entrypoints.llm.LLM`.

Mirrors the upstream interface: `generate` returns, per prompt, a list of per-token
results ending in EOS, and text is recovered with `get_tokenizer().decode(...)`.
Completions describe the engine that produced them, so tests can check sharding,
device pinning, and process reuse. Set `STUB_FAIL_RANK` to make one engine fail to load.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any

EOS = 256  # outside the byte range used for ordinary tokens


@dataclass
class Token:
    token_id: int


class Tokenizer:
    def decode(self, ids: list[int], skip_special_tokens: bool = False, **_: Any) -> str:
        data = bytes(i for i in ids if i < EOS)
        text = data.decode("utf-8")
        return text if skip_special_tokens or EOS not in ids else text + "<eos>"


class LLM:
    def __init__(
        self,
        model: str,
        expert_parallel_size: int,
        context_parallel_size: int,
        engine_rank: int,
        data_parallel_size: int,
        **kwargs: Any,
    ) -> None:
        if not os.path.isfile(os.path.join(model, "config.json")):
            raise FileNotFoundError(f"no config.json in {model}")
        if os.environ.get("STUB_FAIL_RANK") == str(engine_rank):
            raise RuntimeError(f"stub engine {engine_rank} failed to load")
        self.rank = engine_rank
        self.ep = expert_parallel_size
        self.cp = context_parallel_size
        self.dp = data_parallel_size
        self.devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.kwargs = kwargs

    def get_tokenizer(self) -> Tokenizer:
        return Tokenizer()

    def generate(self, prompts: list[str], sampling_params: list[Any], use_tqdm: bool = True, **_: Any) -> list:
        assert len(prompts) == len(sampling_params), "one SamplingParams per prompt"
        return [
            [Token(b) for b in self._complete(p, sp).encode("utf-8")] + [Token(EOS)]
            for p, sp in zip(prompts, sampling_params)
        ]

    def close(self) -> None:
        pass

    def _complete(self, prompt: str, params: Any) -> str:
        so = params.so_args
        if so and so["schema_type"] == "json":
            return json.dumps(_fake(json.loads(so["schema"])))
        if so and so["schema_type"] == "regex":
            # An alternation like "(a|b|c)" answers with its first option; anything else
            # gets a fixed string that matches the examples' "[A-Z]{2}-[0-9]{3}".
            alternation = re.fullmatch(r"\(?([\w ]+)(\|[\w ]+)+\)?", so["schema"])
            return alternation.group(1) if alternation else "AB-123"
        if so and so["schema_type"] == "ebnf":
            return "answer=yes;level=1"
        if prompt.startswith("__env__ "):
            return os.environ.get(prompt.split(" ", 1)[1], "<unset>")
        return (
            f"rank={self.rank} devices={self.devices} ep={self.ep} cp={self.cp} dp={self.dp} "
            f"pid={os.getpid()} max_tokens={params.max_tokens} temperature={params.temperature} | {prompt}"
        )


def _fake(schema: dict, root: dict | None = None) -> Any:
    """A minimal instance of a JSON schema (handles what Pydantic emits)."""
    root = root or schema
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return _fake(root.get("$defs", {})[name], root)
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    if "anyOf" in schema:
        return _fake(schema["anyOf"][0], root)
    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties", {})
        return {k: _fake(v, root) for k, v in props.items()}
    if kind == "array":
        return [_fake(schema.get("items", {}), root)]
    return {"string": "stub", "integer": 1, "number": 1.0, "boolean": True, "null": None}.get(kind, "stub")
