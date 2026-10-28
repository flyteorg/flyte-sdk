"""Structured output: constrain DeepInfer to a Pydantic model's schema, or to a regex.

DeepInfer can constrain decoding to a JSON schema, a regular expression, or an EBNF
grammar, so every completion is guaranteed to parse:

- `extract_tickets` passes a Pydantic model as `json_schema` and gets back validated
  model instances, which Flyte serializes natively as the task output.
- `classify_sentiment` constrains each answer to one of three labels with `regex`.

Both tasks share one warm engine pool on a reusable container.

    flyte run examples/deepinfer_structured_extraction.py extract_tickets \\
        --emails '["My invoice #4411 was charged twice, please refund!", "How do I reset my password?"]'
"""

from typing import Literal, Optional

import flyte
from pydantic import BaseModel

from flyteplugins.deepinfer import DeepInfer, deepinfer_image

# --- Knobs ----------------------------------------------------------------------------
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
GPU = "B200:8"

env = flyte.TaskEnvironment(
    name="deepinfer-structured",
    resources=flyte.Resources(gpu=GPU),
    image=deepinfer_image().with_pip_packages("unionai-reuse"),
    reusable=flyte.ReusePolicy(replicas=(1, 2), idle_ttl=600),
)

llm = DeepInfer(MODEL)


class SupportTicket(BaseModel):
    category: Literal["billing", "account", "bug", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str
    order_id: Optional[str] = None


@env.task
async def extract_tickets(emails: list[str]) -> list[SupportTicket]:
    prompts = [f"Extract a support ticket from this customer email as JSON.\n\nEmail:\n{e}" for e in emails]
    return await llm.generate.aio(prompts, json_schema=SupportTicket, temperature=0, max_tokens=256)


@env.task
async def classify_sentiment(reviews: list[str]) -> list[str]:
    prompts = [f"Is this review positive, negative, or neutral?\n\nReview: {r}\nAnswer:" for r in reviews]
    return await llm.generate.aio(prompts, regex=r"(positive|negative|neutral)", temperature=0, max_tokens=4)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        extract_tickets,
        emails=["My invoice #4411 was charged twice, please refund!", "How do I reset my password?"],
    )
    print("Run URL:", run.url)
