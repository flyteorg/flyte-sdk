"""Examples of events with markdown prompts and webhook integration."""

import time

import flyte

env = flyte.TaskEnvironment(name="events")


@env.task
async def next_task(value: int) -> int:
    return value + 1


# ---------------------------------------------------------------------------
# 1. Basic event with a plain-text prompt
# ---------------------------------------------------------------------------
@env.task
async def basic_event_task(x: int) -> int:
    event = await flyte.new_event.aio(
        "approval",
        prompt="Is it ok to continue?",
        data_type=bool,
    )
    if await event.wait.aio():
        return x + 1
    return -1


# ---------------------------------------------------------------------------
# 2. Markdown prompt – renders rich text in the TUI / UI
# ---------------------------------------------------------------------------
@env.task
async def markdown_event_task(x: int) -> int:
    event = await flyte.new_event.aio(
        "review",
        prompt=(
            "## Review needed\n\n"
            "The pipeline produced the following results:\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            "| Accuracy | 0.95 |\n"
            "| Loss | 0.05 |\n\n"
            "**Should we proceed to deployment?**"
        ),
        prompt_type="markdown",
        data_type=bool,
    )
    approved = await event.wait.aio()
    if approved:
        return await next_task(x)
    return -1


# ---------------------------------------------------------------------------
# 3. Webhook – notify an external system when the event is created.
#    The external system can then POST back to {callback_uri} to signal it.
# ---------------------------------------------------------------------------
@env.task
async def webhook_outbound_task(x: int) -> int:
    event = await flyte.new_event.aio(
        "external_approval",
        prompt="Waiting for external approval via webhook…",
        data_type=bool,
        webhook=flyte.EventWebhook(
            url="https://example.com/hook",
            payload={
                "callback": "{callback_uri}",
                "event": "approval_needed",
                "message": "Pipeline is waiting for approval",
            },
        ),
    )
    # The backend POSTs to the webhook URL with the payload above.
    # An external service can then POST to the {callback_uri} to signal the event.
    approved = await event.wait.aio()
    if approved:
        return x + 1
    return -1


# ---------------------------------------------------------------------------
# 4. Webhook – signal an event programmatically from a remote client
#    (e.g. from a webhook handler, a script, or a CI job)
# ---------------------------------------------------------------------------
def signal_event_remotely(run_name: str, action_name: str):
    """Example of signaling an event from outside the workflow."""
    import flyte.remote as remote

    flyte.init()
    # Poll until the event is available
    while not (event := remote.Event.get("external_approval", run_name=run_name, action_name=action_name)):
        time.sleep(5)

    # Signal the event – the waiting task resumes immediately
    event.signal(True)


# ---------------------------------------------------------------------------
# Main – run the markdown example locally
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    flyte.init()

    r = flyte.run(markdown_event_task, x=10)
    print(r.url)
    print(r.outputs())
