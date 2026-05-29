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
# 4. Webhook - signal an event programmatically from a remote client
#    (e.g. from a webhook handler, a script, or a CI job)
# ---------------------------------------------------------------------------
def signal_event_remotely(run_name: str, action_name: str):
    """Example of signaling an event from outside the workflow."""
    import flyte.remote as remote

    flyte.init()
    # Poll until the event is available
    while not (event := remote.Event.get("external_approval", run_name=run_name, action_name=action_name)):
        time.sleep(5)

    # Signal the event - the waiting task resumes immediately
    event.signal(True)


# ---------------------------------------------------------------------------
# 5. Signal a specific waiting action on a remote run (by console URL)
# ---------------------------------------------------------------------------
# Target action, taken from the console URL:
#   https://dogfood.cloud-staging.union.ai/v2/domain/development/project/flytesnacks/
#       runs/uvnqfmb2nhbtwkkttplt?i=2zvbu2790j80vmmugrco27zi7
RUN_NAME = "uvnqfmb2nhbtwkkttplt"
ACTION_ID = "2zvbu2790j80vmmugrco27zi7"  # the `i=` action highlighted in the URL
# Value delivered to the event. The example tasks declare `data_type=bool`, so
# True == approve / proceed. Change this (or its type) to match your event.
PAYLOAD = True


def signal_remote_action(run_name: str, action_id: str, payload):
    """Signal the event backing ``action_id`` within ``run_name``."""
    import flyte.remote as remote

    # Point at the same project/domain shown in the console URL (endpoint/org/auth
    # still come from your local config).
    flyte.init_from_config(project="flytesnacks", domain="development")

    # The `i=` value identifies a specific action node. Find the event that is either
    # that condition action itself, or whose parent is that action.
    for event in remote.Event.listall(run_name=run_name):
        if event.action_name == action_id or event.pb2.metadata.parent == action_id:
            print(f"Signaling event '{event.name}' (action {event.action_name}) with {payload!r}")
            event.signal(payload)
            print("Signal delivered; the waiting task will resume.")
            return event

    raise RuntimeError(
        f"No waiting event found for action '{action_id}' in run '{run_name}'. "
        "Confirm the run is still paused on an event and that the project/domain are correct."
    )


# ---------------------------------------------------------------------------
# Main - trigger markdown_event_task and signal its "review" event
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import flyte.remote as remote

    flyte.init_from_config()

    # 1. Launch the run. markdown_event_task pauses on its "review" event.
    r = flyte.run(markdown_event_task, x=10)
    print("run url:", r.url)

    # 2. Poll until a condition event shows up on the run, then signal it.
    #    We don't match on a hard-coded name: depending on the backend, the listed
    #    event's name may be the declared name ("review") or its generated action id.
    #    So we list every condition event and pick the one that's waiting.
    EVENT_NAME = "review"  # the name declared in markdown_event_task
    event = None
    while event is None:
        events = list(remote.Event.listall(run_name=r.name))
        if events:
            print("found events:", [(e.name, e.action_name, e.phase) for e in events])
            # Prefer an exact name match; otherwise fall back to the only/first event.
            event = next((e for e in events if e.name == EVENT_NAME), events[0])
        else:
            print("waiting for a condition event to be created…")
            time.sleep(5)

    print(f"signaling event '{event.name}' (action {event.action_name}, phase {event.phase}) with True")
    event.signal(True)

    # 3. The task resumes; print its final output.
    print("outputs:", r.outputs())
