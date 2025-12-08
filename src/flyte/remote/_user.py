from __future__ import annotations

from dataclasses import dataclass

from flyteidl.service import identity_pb2
from flyteidl.service.identity_pb2 import UserInfoResponse

from .._initialize import ensure_client, get_client
from ..syncify import syncify
from ._common import ToJSONMixin


@dataclass
class User(ToJSONMixin):
    pb2: UserInfoResponse

    @syncify
    @classmethod
    async def get(cls) -> User:
        """
        Fetches information about the currently logged in user.
        Returns: A User object containing details about the user.
        """
        ensure_client()

        resp = await get_client().identity_service.UserInfo(identity_pb2.UserInfoRequest())
        return cls(resp)

    def subject(self) -> str:
        return self.pb2.subject

    def name(self) -> str:
        return self.pb2.name
