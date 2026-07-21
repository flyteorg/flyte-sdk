from __future__ import annotations

from typing import Any, cast

from flyteidl2.actions.actions_service_connect import ActionsServiceClient
from flyteidl2.workflow.queue_service_connect import QueueServiceClient
from flyteidl2.workflow.state_service_connect import StateServiceClient

from flyte.remote._client.auth._session import SessionConfig, create_session_config

from ._service_protocol import ActionsService, QueueService, StateService, use_actions_service


class _SwappableHTTPClient:
    """HTTP client facade whose underlying client (and its connection pool) can be replaced.

    connectrpc service clients accept an ``http_client`` at construction and invoke a request
    method on it per RPC. Handing them this stable facade lets the controller abandon a
    suspected-dead connection pool without reaching into connectrpc internals: ``replace``
    installs a fresh client for future requests, while requests and streams already in flight
    keep the old client alive until they finish.

    Methods delegate explicitly (rather than via ``__getattr__``) so that even a bound method
    captured by a caller resolves the current client at call time.
    """

    def __init__(self, inner: Any):
        self._inner = inner

    def replace(self, inner: Any) -> None:
        self._inner = inner

    def delete(self, *args, **kwargs):
        return self._inner.delete(*args, **kwargs)

    def execute(self, *args, **kwargs):
        return self._inner.execute(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self._inner.get(*args, **kwargs)

    def head(self, *args, **kwargs):
        return self._inner.head(*args, **kwargs)

    def options(self, *args, **kwargs):
        return self._inner.options(*args, **kwargs)

    def patch(self, *args, **kwargs):
        return self._inner.patch(*args, **kwargs)

    def post(self, *args, **kwargs):
        return self._inner.post(*args, **kwargs)

    def put(self, *args, **kwargs):
        return self._inner.put(*args, **kwargs)

    def stream(self, *args, **kwargs):
        return self._inner.stream(*args, **kwargs)


class ControllerClient:
    """
    A client for the Controller API.
    """

    def __init__(self, session_cfg: SessionConfig):
        self._session_cfg = session_cfg
        self._http_client = _SwappableHTTPClient(session_cfg.http_client)
        shared: dict[str, Any] = {**session_cfg.connect_kwargs(), "http_client": self._http_client}
        self._state_service = StateServiceClient(**shared)
        self._queue_service = QueueServiceClient(**shared)
        self._actions_service = ActionsServiceClient(**shared) if use_actions_service() else None

    def replace_http_client(self) -> None:
        """Abandon pooled connections by installing a fresh HTTP client behind the facade.

        Used when the shared connection pool is suspected dead (e.g. a black-holed TCP flow
        that times out every request without ever surfacing a transport error). The
        state/queue/actions service clients — and the informers, which share those same
        objects — all send requests through the one facade, so the replacement takes effect
        everywhere on their next call.
        """
        self._http_client.replace(self._session_cfg.new_http_client())

    @classmethod
    async def for_endpoint(cls, endpoint: str, insecure: bool = False, **kwargs) -> ControllerClient:
        session_cfg = await create_session_config(endpoint, None, insecure=insecure, **kwargs)
        return cls(session_cfg)

    @classmethod
    async def for_api_key(cls, api_key: str, insecure: bool = False, **kwargs) -> ControllerClient:
        session_cfg = await create_session_config(None, api_key, insecure=insecure, **kwargs)
        return cls(session_cfg)

    @property
    def state_service(self) -> StateService:
        """
        The state service.
        """
        # The generated connect client matches the protocol at runtime (parameter names and
        # streaming return types differ from the hand-written protocol declarations).
        return cast(StateService, self._state_service)

    @property
    def queue_service(self) -> QueueService:
        """
        The queue service.
        """
        return cast(QueueService, self._queue_service)

    @property
    def actions_service(self) -> ActionsService | None:
        """
        The unified actions service (replaces QueueService + StateService when available).
        """
        return cast("ActionsService | None", self._actions_service)
