from typing import Any, AsyncIterator, Optional

from litellm import acompletion, completion

import flyte


@flyte.trace
async def asingle_shot_llm_call(
    model: str,
    system_prompt: str,
    message: str,
    response_format: Optional[dict[str, str | dict[str, Any]]] = None,
    max_completion_tokens: int | None = None,
) -> AsyncIterator[str]:
    stream = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        response_format=response_format,
        # NOTE: max_token is deprecated per OpenAI API docs, use max_completion_tokens instead if possible
        # NOTE: max_completion_tokens is not currently supported by Together AI, so we use max_tokens instead
        max_tokens=max_completion_tokens,
        timeout=600,
        stream=True,
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.get("content", "")
        if content:
            yield content


@flyte.trace
def single_shot_llm_call(
    model: str,
    system_prompt: str,
    message: str,
    response_format: Optional[dict[str, str | dict[str, Any]]] = None,
    max_completion_tokens: int | None = None,
) -> str:
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        response_format=response_format,
        # NOTE: max_token is deprecated per OpenAI API docs, use max_completion_tokens instead if possible
        # NOTE: max_completion_tokens is not currently supported by Together AI, so we use max_tokens instead
        max_tokens=max_completion_tokens,
        timeout=600,
    )
    return response.choices[0].message["content"]  # type: ignore
