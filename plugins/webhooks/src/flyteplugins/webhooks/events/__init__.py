"""Typed constants for the webhook events each provider can send.

Register handlers with these instead of raw strings, so an editor can complete
them and a typo fails at import rather than by silently never matching:

```python
from flyteplugins.webhooks import events

@app_env.on_event(events.github.PullRequest.OPENED)
async def triage(event): ...
```

Each provider is a submodule, and within it each class is one event type. Where
a provider splits type and action, members spell the `type.action` pattern
`on_event` matches on and `ANY` is the bare type, matching every action:

```python
@app_env.on_event(events.github.PullRequest.ANY)      # every pull_request action
@app_env.on_event(events.github.PullRequest.OPENED)   # just pull_request.opened
```

These are `str` subclasses, so they are drop-in wherever a pattern string is
accepted. `on_event` still takes plain strings too — reach for one when a
provider ships an event these constants do not cover yet.
"""

from . import clickup, github, jira, linear, slack
from ._base import EventType

__all__ = ["EventType", "clickup", "github", "jira", "linear", "slack"]
