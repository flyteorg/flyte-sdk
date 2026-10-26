"""Structured extraction: constrain DeepInfer's output to a Pydantic model's JSON schema.

DeepInfer can constrain decoding to a regex, an EBNF grammar, or a JSON schema, so every
completion is guaranteed to parse. Here a Pydantic model defines the schema, the task
returns validated model instances (which Flyte serializes natively), and the engine is
kept warm on a reusable container so a stream of small extraction jobs pays the model
load once.

The engine spans all of the node's GPUs as a single context-parallel engine
(`cp_size` = GPU count), which can live in the task process and therefore be cached
across invocations. For maximum throughput on one large batch, prefer one engine per
GPU (`deepinfer_data_parallel.py`).

    flyte run examples/deepinfer_structured_extraction.py extract_tickets \\
        --emails '["My invoice #4411 was charged twice, please refund!", "How do I reset my password?"]'
"""

import functools
import json
from typing import Literal

import flyte
from pydantic import BaseModel

from flyteplugins.deepinfer import (
    create_llm,
    deepinfer_image_from_source,
    get_actual_num_gpus,
    sampling_params,
    setup_common_env,
    structured_output,
)

env = flyte.TaskEnvironment(
    name="deepinfer-structured-extraction",
    reusable=flyte.ReusePolicy(replicas=(1, 2), idle_ttl=600),
    resources=flyte.Resources(gpu=8),
    image=deepinfer_image_from_source(),
)


class SupportTicket(BaseModel):
    category: Literal["billing", "account", "bug", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str
    order_id: str | None = None


PROMPT = """Extract a support ticket from the customer email below. Respond with JSON only.

Email:
{email}
"""


@functools.lru_cache(maxsize=1)
def _engine(model: str):
    # Cached per warm container: weights and CUDA warm-up are paid once.
    setup_common_env()
    return create_llm(model, cp_size=get_actual_num_gpus())


@env.task
async def extract_tickets(
    emails: list[str],
    model: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    max_tokens: int = 256,
) -> list[SupportTicket]:
    from deepinfer.sampling_params import SamplingParams

    llm = _engine(model)
    so_args = structured_output("json", json.dumps(SupportTicket.model_json_schema()))
    params = SamplingParams(**sampling_params(temperature=0, max_tokens=max_tokens, so_args=so_args))
    outputs = llm.generate(
        prompts=[PROMPT.format(email=email) for email in emails],
        sampling_params=[params] * len(emails),
        use_tqdm=False,
    )
    tokenizer = llm.get_tokenizer()
    return [
        SupportTicket.model_validate_json(tokenizer.decode([tok.token_id for tok in output], skip_special_tokens=True))
        for output in outputs
    ]


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        extract_tickets,
        emails=[
            "My invoice #4411 was charged twice, please refund!",
            "How do I reset my password?",
        ],
    )
    print("Run URL:", run.url)
