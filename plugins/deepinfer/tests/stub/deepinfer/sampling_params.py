"""Stand-in for `deepinfer.sampling_params`."""

from typing import Any, Optional


class SamplingParams:
    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 16,
        ignore_eos: bool = False,
        top_logprobs: int = 0,
        so_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.top_logprobs = top_logprobs
        self.so_args = so_args
