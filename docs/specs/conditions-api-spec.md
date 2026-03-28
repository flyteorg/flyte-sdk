# Conditions (Events) API Spec

**Status:** Draft
**Author:** Ketan Umare
**Date:** 2026-03-27

---

## Overview

Conditions (also called Events) allow a running task to pause execution and wait for an external signal before continuing. This enables human-in-the-loop approvals, external system callbacks, and cross-workflow coordination.

This spec covers:
1. The SDK-side Conditions API (already partially implemented)
2. The proto changes required for the backend
3. The remote controller implementation
4. The webhook/callback mechanism for external signaling
5. Use cases

---

## Use Cases

### 1. Human Approval Gate

A data pipeline that processes sensitive financial data requires a human to approve before the final write to production.

```python
@env.task
async def etl_pipeline():
    staged_data = transform(raw_data)

    approval = await flyte.new_event(
        name="prod_write_approval",
        prompt="Approve writing staged data to production?",
        data_type=bool,
        timeout=timedelta(hours=24),
    )
    approved = await approval.wait()

    if approved:
        write_to_prod(staged_data)
    else:
        raise RuntimeError("Pipeline rejected by reviewer")
```

The reviewer signals the event via the UI, CLI, or the remote API (`flyte.remote.Event.signal()`).

### 2. Webhook Callback from an External System

An ML training pipeline kicks off a model evaluation in an external service. The external service calls back when evaluation is complete.

```python
@env.task
async def train_and_evaluate():
    model = train_model()

    eval_event = await flyte.new_event(
        name="evaluation_complete",
        prompt="Waiting for external evaluation...",
        data_type=str,
        timeout=timedelta(hours=2),
        webhook=flyte.EventWebhook(
            url="https://eval-service.internal/evaluate",
            payload={
                "model_id": model.id,
                "callback": "{callback_uri}",
            },
        ),
    )

    eval_result = await eval_event.wait()
    # eval_result is the string the external service sent back
```

When the event is registered, the backend POSTs to the webhook URL with the payload. The `{callback_uri}` template variable is replaced with the actual URI that the external service should call to signal the event. The external service later POSTs to that callback URI with the payload value to unblock the task.

### 3. Cross-Workflow Coordination

A deployment workflow waits for a separate CI workflow to signal that tests have passed.

```python
@env.task
async def deploy_service():
    event = await flyte.new_event(
        name="ci_passed",
        prompt="Waiting for CI pipeline to pass...",
        data_type=bool,
        timeout=timedelta(minutes=30),
    )

    passed = await event.wait()
    if passed:
        deploy()
```

The CI workflow signals this event using the remote API:

```python
event = flyte.remote.Event.get("ci_passed", run_name="deploy-run-123", action_name="deploy_service")
event.signal(True)
```

### 4. Parameterized Manual Input

A task that generates a report pauses to collect a user-supplied threshold value before proceeding.

```python
@env.task
async def generate_report():
    raw_stats = compute_stats()

    threshold_event = await flyte.new_event(
        name="threshold",
        prompt="Enter the anomaly threshold (0.0 - 1.0):",
        data_type=float,
        description="Values above this threshold will be flagged as anomalies in the report.",
    )

    threshold = await threshold_event.wait()
    report = build_report(raw_stats, threshold=threshold)
    return report
```

### 5. Webhook to Slack for Approval via Bot

A task sends a Slack message with an approve/reject button. The Slack bot calls the callback URI when a user clicks.

```python
@env.task
async def release_gate():
    event = await flyte.new_event(
        name="release_approval",
        prompt="Release approval required",
        data_type=bool,
        timeout=timedelta(hours=4),
        webhook=flyte.EventWebhook(
            url="https://slack-bot.internal/approval-request",
            payload={
                "channel": "#releases",
                "message": "Release v2.1 requires approval",
                "approve_callback": "{callback_uri}",
            },
        ),
    )

    approved = await event.wait()
    if not approved:
        raise RuntimeError("Release rejected")
```

### 6. Timeout with Fallback

An event with a timeout that falls back to a default action if no signal is received.

```python
@env.task
async def optional_review():
    event = await flyte.new_event(
        name="review",
        prompt="Review output?",
        data_type=bool,
        timeout=timedelta(minutes=10),
    )

    try:
        reviewed = await event.wait()
    except flyte.errors.EventTimedoutError:
        # No reviewer responded; proceed with default
        reviewed = True

    if reviewed:
        publish()
```

---

## SDK API

### `flyte.EventWebhook`

```python
@dataclass
class EventWebhook:
    url: str
    payload: Optional[Dict[str, Any]] = None
```

- `url` — The HTTP endpoint to POST to when the event is registered.
- `payload` — Optional JSON body. All string values are recursively scanned for `{callback_uri}`, which the backend substitutes with the URI that can be used to signal the event.

### `flyte.new_event()`

```python
async def new_event(
    name: str,
    /,
    prompt: str = "Approve?",
    prompt_type: Literal["text", "markdown"] = "text",
    data_type: Type[bool | int | float | str] = bool,
    description: str = "",
    timeout: timedelta | int | float | None = None,
    webhook: EventWebhook | None = None,
) -> _Event
```

Creates and registers an event. In a task context, this calls `controller.register_event(event)` which persists the event on the backend.

### `_Event.wait()`

```python
async def wait(self) -> EventType
```

Blocks the task until the event is signaled. Raises `flyte.errors.EventTimedoutError` if the timeout expires.

### `flyte.remote.Event` (Client API)

Already implemented. Provides `get()`, `listall()`, and `signal()` for external consumers to interact with events.

---

## Proto Changes

### 1. No `CreateEvent` RPC — Conditions are enqueued as actions

There is **no separate `CreateEvent` RPC**. Events (conditions) are created dynamically by the controller when the task calls `flyte.new_event(...)`. The controller enqueues a `ConditionAction` via `ActionsService.Enqueue`, which already supports conditions as a first-class action type.

The relevant protos are:

- [`ActionsService`](https://github.com/flyteorg/flyte/blob/v2/flyteidl2/actions/actions_service.proto) — the unified service for enqueuing and managing actions
- [`ConditionAction`](https://github.com/flyteorg/flyte/blob/v2/flyteidl2/workflow/run_definition.proto) — the condition spec within `run_definition.proto`
- [`QueueService.EnqueueAction`](https://github.com/flyteorg/flyte/blob/v2/flyteidl2/workflow/queue_service.proto) — the legacy queue service (being replaced by `ActionsService`)

### 2. `ConditionAction` message (existing)

The `ConditionAction` message already exists in [`run_definition.proto`](https://github.com/flyteorg/flyte/blob/v2/flyteidl2/workflow/run_definition.proto):

```protobuf
// ConditionAction is used to define a condition that can be evaluated at runtime. It can be used to
// await a signal from an external system and can carry a value.
message ConditionAction {
  // Name is the unique identifier for the action. It must be unique within the defined scope below.
  string name = 1;

  oneof scope {
    // RunId is the unique identifier for the run this action is associated with.
    string run_id = 2;
    // ActionId is the unique identifier for the action this action is associated with.
    string action_id = 3;
    // Global indicates the condition is global and can be used across all runs and actions.
    bool global = 4;
  }

  // Type is the type of the value the condition is expected.
  flyteidl2.core.LiteralType type = 6;

  // Prompt is the prompt that will be shown to the user when the condition is awaited.
  string prompt = 7;

  // Description of the condition.
  string description = 8;
}
```

### 3. `ActionsService.Enqueue` (existing)

The [`ActionsService.Enqueue`](https://github.com/flyteorg/flyte/blob/v2/flyteidl2/actions/actions_service.proto) RPC already accepts `ConditionAction` in the `Action.spec` oneof:

```protobuf
message Action {
  flyteidl2.common.ActionIdentifier action_id = 1;
  optional string parent_action_name = 2;
  string input_uri = 3;
  string run_output_base = 4;
  string group = 5;
  string subject = 6;

  oneof spec {
    flyteidl2.workflow.TaskAction task = 7;
    flyteidl2.workflow.TraceAction trace = 8;
    flyteidl2.workflow.ConditionAction condition = 9;  // <-- used for events
  }
}

message EnqueueRequest {
  Action action = 1;
  flyteidl2.task.RunSpec run_spec = 2;
}
```

### 4. Proposed additions to `ConditionAction`

The existing `ConditionAction` needs a few additions to support the full event feature set:

```protobuf
message ConditionAction {
  string name = 1;

  oneof scope { ... }  // existing

  flyteidl2.core.LiteralType type = 6;   // existing
  string prompt = 7;                      // existing
  string description = 8;                 // existing

  // --- New fields ---

  // How to render the prompt.
  ConditionPromptType prompt_type = 9;

  // Optional timeout. If not signaled within this duration, the condition
  // transitions to a timed-out state.
  google.protobuf.Duration timeout = 10;

  // Optional webhook to fire when the condition action is created.
  ConditionWebhook webhook = 11;
}

enum ConditionPromptType {
  CONDITION_PROMPT_TYPE_UNSPECIFIED = 0;
  CONDITION_PROMPT_TYPE_TEXT = 1;
  CONDITION_PROMPT_TYPE_MARKDOWN = 2;
}

// Webhook configuration for condition notification.
// When present, the backend POSTs to the URL upon condition creation.
message ConditionWebhook {
  // The URL to POST to.
  string url = 1;

  // Optional JSON payload as a Struct.
  // String values may contain the template variable "{callback_uri}"
  // which the backend replaces with the actual signal URI for this condition.
  google.protobuf.Struct payload = 2;
}
```

### 5. How it works — the flow

1. Task code calls `flyte.new_event(...)`.
2. The SDK's `RemoteController.register_event()` builds a `ConditionAction` and enqueues it via `ActionsService.Enqueue`.
3. The backend creates the condition action. If a webhook is configured, the backend fires it (see Callback Mechanism below).
4. Task code calls `event.wait()`, which uses `ActionsService.WatchForUpdates` (or polls `GetLatestState`) until the condition is signaled.
5. An external system (or user via UI/CLI) signals the condition, which completes the action.

This keeps event creation as part of the existing action lifecycle — no new service or RPC needed.

---

## Remote Controller Implementation

### `register_event`

The `RemoteController.register_event()` method (currently `NotImplementedError`) will:

1. Build a `ConditionAction` from the `_Event` object.
2. Enqueue it as an action via `ActionsService.Enqueue`.
3. If the event has a webhook, the backend handles firing it (see Callback Mechanism below).

```python
async def register_event(self, event: Any):
    from flyte._event import _Event

    if not isinstance(event, _Event):
        raise TypeError(f"Expected _Event, got {type(event)}")

    parent_action_id = self._get_current_action_id()

    # Map SDK types to proto LiteralType
    type_map = {
        bool: core_types_pb2.LiteralType(simple=core_types_pb2.SIMPLE_BOOLEAN),
        int: core_types_pb2.LiteralType(simple=core_types_pb2.SIMPLE_INTEGER),
        float: core_types_pb2.LiteralType(simple=core_types_pb2.SIMPLE_FLOAT),
        str: core_types_pb2.LiteralType(simple=core_types_pb2.SIMPLE_STRING),
    }
    prompt_type_map = {
        "text": run_definition_pb2.CONDITION_PROMPT_TYPE_TEXT,
        "markdown": run_definition_pb2.CONDITION_PROMPT_TYPE_MARKDOWN,
    }

    webhook_pb = None
    if event.webhook is not None:
        webhook_pb = run_definition_pb2.ConditionWebhook(
            url=event.webhook.url,
            payload=struct_pb2.Struct(fields=event.webhook.payload) if event.webhook.payload else None,
        )

    timeout_pb = None
    if event._timeout_seconds is not None:
        timeout_pb = duration_pb2.Duration(seconds=int(event._timeout_seconds))

    condition = run_definition_pb2.ConditionAction(
        name=event.name,
        action_id=parent_action_id.action_name,  # scope to current action
        type=type_map[event.data_type],
        prompt=event.prompt,
        description=event.description,
        prompt_type=prompt_type_map[event.prompt_type],
        timeout=timeout_pb,
        webhook=webhook_pb,
    )

    # Generate a unique action name for this condition
    condition_action_id = common_pb2.ActionIdentifier(
        org=parent_action_id.org,
        project=parent_action_id.project,
        domain=parent_action_id.domain,
        run_name=parent_action_id.run_name,
        action_name=f"{parent_action_id.action_name}.{event.name}",
    )

    action = actions_service_pb2.Action(
        action_id=condition_action_id,
        parent_action_name=parent_action_id.action_name,
        input_uri=self._run_output_base,  # conditions don't need separate input
        run_output_base=self._run_output_base,
        condition=condition,
    )

    await self._actions_service.Enqueue(
        actions_service_pb2.EnqueueRequest(action=action)
    )
```

### `wait_for_event`

The `RemoteController.wait_for_event()` method watches for updates to the condition action until it is signaled or the timeout expires.

```python
async def wait_for_event(self, event: Any) -> Any:
    from flyte._event import _Event

    if not isinstance(event, _Event):
        raise TypeError(f"Expected _Event, got {type(event)}")

    parent_action_id = self._get_current_action_id()
    condition_action_id = common_pb2.ActionIdentifier(
        org=parent_action_id.org,
        project=parent_action_id.project,
        domain=parent_action_id.domain,
        run_name=parent_action_id.run_name,
        action_name=f"{parent_action_id.action_name}.{event.name}",
    )

    deadline = None
    if event._timeout_seconds is not None:
        deadline = time.monotonic() + event._timeout_seconds

    # Use WatchForUpdates to stream condition action state changes
    async for response in self._actions_service.WatchForUpdates(
        actions_service_pb2.WatchForUpdatesRequest(
            parent_action_id=condition_action_id,
        )
    ):
        if response.HasField("action_update"):
            update = response.action_update
            if update.status.phase == common_pb2.ACTION_PHASE_SUCCEEDED:
                return _decode_payload(update, event.data_type)

        if deadline is not None and time.monotonic() >= deadline:
            raise flyte.errors.EventTimedoutError(
                f"Event '{event.name}' was not signaled within {event._timeout_seconds} seconds."
            )

    # Fallback: if stream ends without signal, raise timeout
    raise flyte.errors.EventTimedoutError(
        f"Event '{event.name}' stream ended without being signaled."
    )
```

> **Note:** `WatchForUpdates` provides at-least-once delivery semantics. A simpler polling fallback using `GetLatestState` can be used if the stream is unavailable.

---

## Callback Mechanism (Webhook Firing)

### Summary

When a condition has a `webhook` configured, the backend must fire it after the condition action is created. The webhook payload's `{callback_uri}` template is replaced with a URI that external systems can call to signal the condition.

### How the callback URI is constructed

The backend constructs the callback URI as a fully qualified endpoint for signaling the condition action:

```
https://<flyte-api-host>/api/v2/actions/<org>/<project>/<domain>/<run_name>/<condition_action_name>/signal
```

Or, alternatively, a short-lived signed URL if authentication is required for external callers.

### How the callback is fired

**The webhook will be fired using the existing notifications engine and event system.** The notifications engine (`flyte.notify`) already supports:

- HTTP webhooks with template variable substitution
- Async `httpx`-based delivery
- Error handling with logging (never crashes the run)

The backend will extend the notifications delivery pipeline to support condition-webhook dispatch as a new delivery trigger. This keeps the implementation DRY and leverages the battle-tested notification sender.

**The detailed design of the callback delivery pipeline (backend-side) will be specced separately.** This includes:

- The exact notifications engine integration points
- Authentication and signing for callback URIs
- Retry policy for failed webhook deliveries
- Rate limiting and abuse prevention
- Callback URI expiry and lifecycle management

### Local execution

In local execution mode, the `LocalController` already fires the webhook directly using `httpx.AsyncClient`:

- Substitutes `{callback_uri}` in all string values of the payload
- POSTs to the webhook URL
- Logs failures but does not block the event registration (fire-and-forget)
- Uses a synthetic local callback URI: `local://events/<event_name>/signal`

This is implemented and tested.

---

## Data Flow

### Remote Execution

```
Task Code                  SDK (RemoteController)          Backend (ActionsService)        External System
─────────                  ────────────────────           ──────────────────────         ───────────────
new_event(webhook=wh) ──> register_event(event) ──────> Enqueue(ConditionAction)
                                                           │
                                                           ├─ Create condition action
                                                           ├─ Build callback_uri
                                                           ├─ Substitute {callback_uri} in webhook payload
                                                           └─ Fire webhook ─────────────────────────────> POST url (payload)
                                                                                                           │
event.wait() ─────────> wait_for_event(event) ─────────> WatchForUpdates(condition_action_id)              │
                           │                                 │                                              │
                           │  (blocks on stream)             │                                              │
                           │                                 │                   POST callback_uri(payload) <┘
                           │                              Signal condition ──────────────────────────────────
                           │                                 │
                           │                              action_update ──> SUCCEEDED ──> return payload
                        <──┘ returns payload
```

### Local Execution

```
Task Code                  SDK (LocalController)          External System
─────────                  ──────────────────            ───────────────
new_event(webhook=wh) ──> register_event(event)
                           │
                           ├─ Store in _registered_events
                           ├─ Substitute {callback_uri} in payload
                           └─ POST webhook url (payload) ──────────────> receives webhook
                                                                          │
event.wait() ─────────> wait_for_event(event)                             │
                           │                                              │
                           ├─ (TUI mode) render input panel               │
                           └─ (Console mode) rich prompt                  │
                              user submits value ──> return payload
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Webhook POST fails (network error, 5xx) | Log warning; event is still created. Task blocks on `wait()` as normal. |
| Webhook POST returns 4xx | Log warning; event is still created. |
| Event timeout expires | `wait()` raises `flyte.errors.EventTimedoutError`. |
| Invalid data_type | `_Event.__post_init__` raises `TypeError` at creation time. |
| `signal()` called with wrong type | `signal()` raises `TypeError` on the caller side. |
| Event signaled multiple times | Idempotent — second signal is a no-op (backend enforces). |
| `wait()` called outside task context | Raises `RuntimeError`. |

---

## Open Questions

1. **Streaming vs polling for `wait_for_event`:** The initial implementation uses polling. Should we add a `WatchEvent` streaming RPC to the `EventService` for push-based notification?

2. **Callback URI authentication:** Should callback URIs be signed/time-limited, or rely on network-level auth (e.g., VPC, mTLS)?

3. **Webhook retry policy:** How many retries, what backoff? This will be covered in the callback delivery spec.

4. **Multiple webhooks per event:** Current design supports one webhook per event. Is there a need for multiple?

---

## Out of Scope

- Backend implementation of condition action handling in `ActionsService`
- Backend webhook delivery pipeline (to be specced separately)
- Callback URI signing and authentication
- UI rendering of conditions in the Flyte console
