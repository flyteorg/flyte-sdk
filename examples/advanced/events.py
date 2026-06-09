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
    ev2 = await flyte.new_event.aio(
        "review",
        prompt="Is it really ok to continue, again?",
        data_type=bool,
    )
    if await event.wait.aio():
        if await ev2.wait.aio():
            return x + 1
    return -1


# ---------------------------------------------------------------------------
# 2. Markdown prompt - renders rich text in the TUI / UI
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
# 3. Webhook - notify an external system when the event is created.
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
# 4. Signal an event programmatically from outside the workflow
#    (e.g. from a webhook handler, a script, or a CI job)
# ---------------------------------------------------------------------------
def signal_event_remotely(run_name: str, event_name: str, payload):
    """Signal an event by name within a run.

    Equivalent CLI form once you know the condition action id::

        flyte signal event <run-name> <action-name> <value>
    """
    import flyte.remote as remote

    flyte.init()
    # Poll until the named event appears in the run.
    while not (event := remote.Event.get(event_name, run_name=run_name)):
        time.sleep(5)

    event.signal(payload)


# ---------------------------------------------------------------------------
# Main - trigger markdown_event_task and signal its "review" event
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import flyte.remote as remote

    flyte.init_from_config()

    # 1. Launch the run. markdown_event_task pauses on its "review" event.
    r = flyte.run(markdown_event_task, x=10)
    print("run url:", r.url)

    # 2. Poll until the "review" condition shows up, then signal it with True.
    EVENT_NAME = "review"
    event = None
    while event is None:
        events = list(remote.Event.listall(run_name=r.name))
        if events:
            print("found events:", [(e.name, e.action_name, e.phase) for e in events])
            # Prefer an exact name match; fall back to the only/first event.
            event = next((e for e in events if e.name == EVENT_NAME), events[0])
        else:
            print("waiting for a condition event to be created…")
            time.sleep(5)

    print(f"signaling event '{event.name}' (action {event.action_name}, phase {event.phase}) with True")
    event.signal(True)

    # Equivalent CLI invocation (once you know the action id from the UI/logs):
    #   flyte signal event <r.name> <event.action_name> true

    # 3. The task resumes; print its final output.
    print("outputs:", r.outputs())
