"""Chained conditions (signals) covering every signal input type.

A *condition* (a.k.a. signal/event) pauses a run until an external actor signals
it via the UI, the ``flyte signal event`` CLI, or ``remote.Event.signal(...)``.

This example wires four conditions together so that each one only runs if the
previous one resolved a certain way — a *chain*. It exercises all four payload
types accepted by ``flyte.new_event``:

- ``bool``  → approval gate
- ``int``   → integer quantity (e.g. batch size)
- ``float`` → decimal threshold (e.g. confidence cutoff)
- ``str``   → free-form text (e.g. a release tag)

The first condition also carries a ``timeout``: if no one signals it within the
deadline it auto-fails and ``wait()`` raises ``flyte.errors.EventTimedoutError``,
which the task catches to abort the chain instead of blocking forever.

The final condition carries a ``webhook``: when it is created the backend POSTs
the configured payload to an external URL (substituting ``{callback_uri}``), so an
external system can drive the sign-off instead of a human at the UI/CLI.

Several prompts use ``prompt_type="markdown"`` with embedded HTML (tables, ``<kbd>``,
``<code>``, ``<small>``…). Markdown allows inline HTML, so the web UI renders both;
terminal renderers (CLI/TUI) show the raw HTML tags as plain text.

Run it remotely and signal each condition in turn::

    python chained_conditions.py

Or signal manually once you know the action ids::

    flyte signal event <run-name> <action-name> true
    flyte signal event <run-name> <action-name> 8
    flyte signal event <run-name> <action-name> 0.85
    flyte signal event <run-name> <action-name> v2.1.0
"""

from datetime import timedelta

import flyte
import flyte.errors

env = flyte.TaskEnvironment(name="chained_conditions")


@env.task
async def deploy(tag: str, batch_size: int, threshold: float) -> str:
    return f"deployed {tag} (batch_size={batch_size}, threshold={threshold:.2f})"


@env.task
async def chained_signals() -> str:
    # 1. bool — the gate, with a timeout. If nobody signals within 5 minutes the
    # condition auto-fails and wait() raises EventTimedoutError, so the chain
    # never blocks forever waiting on an absent approver.
    approval = await flyte.new_condition.aio(
        "approval",
        # markdown + embedded HTML: the web UI renders both (terminal renderers
        # fall back to showing the raw HTML tags).
        prompt=(
            "## Deployment approval\n\n"
            "Pipeline **chained_conditions** is ready for review:\n\n"
            "<table>\n"
            "  <tr><th>Stage</th><th>Status</th></tr>\n"
            "  <tr><td>Build</td><td>&#9989; passed</td></tr>\n"
            "  <tr><td>Tests</td><td>&#9989; passed</td></tr>\n"
            "</table>\n\n"
            "> Approve to continue. <em>Auto-expires in 5&nbsp;minutes.</em>"
        ),
        prompt_type="markdown",
        data_type=bool,
        timeout=timedelta(minutes=5),
    )
    try:
        approved = await approval.wait.aio()
    except flyte.errors.ConditionTimedoutError:
        return "aborted: approval timed out (no response within 5 minutes)"
    if not approved:
        return "rejected at approval gate"

    # 2. int — only asked once approved. Chains off the bool above.
    batch_condition = await flyte.new_condition.aio(
        "batch_size",
        prompt="Approved. How many records per batch?",
        data_type=int,
    )
    batch_size = await batch_condition.wait.aio()
    if batch_size <= 0:
        return f"aborted: invalid batch_size {batch_size}"

    # 3. float — only asked once we have a valid batch size. Chains off the int.
    threshold_condition = await flyte.new_condition.aio(
        "threshold",
        # markdown + HTML: bold/inline-code from markdown, <kbd>/<small> from HTML.
        prompt=(
            f"### Confidence threshold\n\n"
            f"Batch size **{batch_size}** accepted. "
            f"Enter the cutoff as a float in <kbd>0.0</kbd>-<kbd>1.0</kbd>.\n\n"
            f"<small>Values outside the range abort the run.</small>"
        ),
        prompt_type="markdown",
        data_type=float,
    )
    threshold = await threshold_condition.wait.aio()
    if not 0.0 <= threshold <= 1.0:
        return f"aborted: threshold {threshold} out of range"

    # 4. str — the final link in the chain, gated on everything above.
    tag_condition = await flyte.new_condition.aio(
        "release_tag",
        # markdown + HTML: heading/list from markdown, <code>/<br> from HTML.
        prompt=(
            "#### Release tag\n\n"
            "All checks passed &#127881;. Enter the tag to deploy, e.g. "
            "<code>v2.1.0</code>.\n\n"
            "- Use semver: `MAJOR.MINOR.PATCH`\n"
            "- Must be unique per release"
        ),
        prompt_type="markdown",
        data_type=str,
    )
    tag = await tag_condition.wait.aio()

    # # 5. webhook — notify an external system when this condit©ion is created. The
    # #    backend POSTs the payload to the URL, substituting {callback_uri} with the
    # #    URI that system can POST back to in order to signal the condition. Chains
    # #    off everything above; deploy only runs once the webhook is signaled True.
    # signoff = await flyte.new_condition.aio(
    #     "external_signoff",
    #     prompt="Waiting for external sign-off via webhook…",
    #     data_type=bool,
    #     webhook=flyte.ConditionWebhook(
    #         url="https://example.com/hook",
    #         payload={
    #             "callback": "{callback_uri}",
    #             "condition": "deploy_signoff",
    #             "tag": tag,
    #         },
    #     ),
    # )
    # if not await signoff.wait.aio():
    #     return "aborted: external sign-off rejected"

    return await deploy(tag=tag, batch_size=batch_size, threshold=threshold)


if __name__ == "__main__":
    import time

    import flyte.remote as remote

    flyte.init_from_config()

    # Launch the run; it pauses on the first ("approval") condition and then, in
    # turn, on each chained condition until every one is signaled.
    r = flyte.run(chained_signals)
    print("run url:", r.url)
    print()
    print("This run is WAITING for you. Signal each condition as it appears,")
    print("either from the UI above or with the printed CLI command.")
    print("Payload types: approval=bool, batch_size=int, threshold=float, release_tag=str")
    print()

    # Poll for conditions that are still waiting and show how to signal them.
    # We do NOT signal them here — the whole point is to wait for *your* input.
    announced: set[str] = set()
    terminal = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"}
    while True:
        for e in remote.Condition.listall(run_name=r.name):
            # Only surface conditions that haven't been signaled yet.
            if e.phase != "RUNNING" or e.name in announced:
                continue
            announced.add(e.name)
            type_name = e.data_type.__name__ if e.data_type else "value"
            print(f"\n[waiting] condition '{e.name}' needs a {type_name}. Signal it with:")
            print(f"    flyte signal condition {r.name} {e.name} <{type_name}>")

        # Re-fetch the run for a fresh phase ('r' holds a launch-time snapshot).
        if remote.Condition.get(name=r.name).phase in terminal:
            break
        time.sleep(5)

    print("\noutputs:", r.outputs())
