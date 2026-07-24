import time

import flyte

env = flyte.TaskEnvironment(name="conditions")


@env.task
async def next_task(value: int) -> int:
    return value + 1


# ---------------------------------------------------------------------------
# 1. Basic condition with a plain-text prompt
# ---------------------------------------------------------------------------
@env.task
async def basic_condition_task(x: int) -> int:
    condition = await flyte.new_condition.aio(
        "approval",
        prompt="Is it ok to continue?",
        data_type=bool,
    )
    cond2 = await flyte.new_condition.aio(
        "review",
        prompt="Is it really ok to continue, again?",
        data_type=bool,
    )
    if await condition.wait.aio():
        if await cond2.wait.aio():
            return x + 1
    return -1


# ---------------------------------------------------------------------------
# 2. Markdown prompt - renders rich text in the TUI / UI
# ---------------------------------------------------------------------------
@env.task
async def markdown_condition_task(x: int) -> int:
    try:
        await t0()  # fails
    except Exception:
        condition = await flyte.new_condition.aio(
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
        approved = await condition.wait.aio()
        if approved:
            return await next_task(x)
        return -1
    return 0


# ---------------------------------------------------------------------------
# 3. Webhook - notify an external system when the condition is created.
#    The external system can then POST back to {callback_uri} to signal it.
# ---------------------------------------------------------------------------
@env.task
async def webhook_outbound_task(x: int) -> int:
    condition = await flyte.new_condition.aio(
        "external_approval",
        prompt="Waiting for external approval via webhook…",
        data_type=bool,
        webhook=flyte.ConditionWebhook(
            url="https://example.com/hook",
            payload={
                "callback": "{callback_uri}",
                "condition": "approval_needed",
                "message": "Pipeline is waiting for approval",
            },
        ),
    )
    # The backend POSTs to the webhook URL with the payload above.
    # An external service can then POST to the {callback_uri} to signal the condition.
    approved = await condition.wait.aio()
    if approved:
        return x + 1
    return -1


# ---------------------------------------------------------------------------
# 4. Signal a condition programmatically from outside the workflow
#    (e.g. from a webhook handler, a script, or a CI job)
# ---------------------------------------------------------------------------
def signal_condition_remotely(run_name: str, condition_name: str, payload):
    """Signal a condition by name within a run.

    Equivalent CLI form once you know the condition action id::

        flyte signal condition <run-name> <action-name> <value>
    """
    import flyte.remote as remote

    flyte.init()
    # Poll until the named condition appears in the run.
    while not (condition := remote.Condition.get(condition_name, run_name=run_name)):
        time.sleep(5)

    condition.signal(payload)


# ---------------------------------------------------------------------------
# Main - trigger markdown_condition_task and signal its "review" condition
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import flyte.remote as remote

    flyte.init_from_config()

    # 1. Launch the run. markdown_condition_task pauses on its "review" condition.
    r = flyte.run(markdown_condition_task, x=10)
    print("run url:", r.url)

    # 2. Poll until the "review" condition shows up, then signal it with True.
    CONDITION_NAME = "review"
    condition = None
    while condition is None:
        conditions = list(remote.Condition.listall(run_name=r.name))
        if conditions:
            print(
                "found conditions:",
                [(c.name, c.action_name, c.phase) for c in conditions],
            )
            # Prefer an exact name match; fall back to the only/first condition.
            condition = next((c for c in conditions if c.name == CONDITION_NAME), conditions[0])
        else:
            print("waiting for a condition to be created…")
            time.sleep(5)

    print(f"signaling condition '{condition.name}' (action {condition.action_name}, phase {condition.phase}) with True")
    condition.signal(True)

    # Equivalent CLI invocation (once you know the action id from the UI/logs):
    #   flyte signal condition <r.name> <condition.action_name> true

    # 3. The task resumes; print its final output.
    print("outputs:", r.outputs())
