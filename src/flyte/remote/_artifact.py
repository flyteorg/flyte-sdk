from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Dict, Literal

from flyte.artifacts import Artifact as CoreArtifact  # Ensure the core Artifact is imported
from flyte.remote._common import ToJSONMixin
from flyte.syncify import syncify


@dataclass
class Artifact(ToJSONMixin):
    """
    A class representing a project in the Union API.
    """

    pb2: Any  # Replace 'Any' with the actual protobuf type when available

    @syncify
    @classmethod
    async def get(cls, name: str, version: str | Literal["latest"] = "latest") -> Artifact:
        """
        Get an artifact by its name and version.

        :param name: The name of the artifact.
        :param version: The version of the artifact.
        """
        raise NotImplementedError("Artifact retrieval not yet implemented.")

    @syncify
    @classmethod
    async def listall(
        cls,
        name: str | None = None,
        created_after: datetime | None = None,
        limit: int = -1,
        **partition_match: Dict[str, str],
    ) -> AsyncIterator[Artifact]:
        """
        List artifacts by name prefix and optional partition match.

        :param name: The name prefix of the artifacts.
        :param created_after: Filter artifacts created after this datetime.
        :param limit: The maximum number of artifacts to return. -1 for no limit.
        :param partition_match: Key-value pairs to filter artifacts by partition.
        :return: A list of artifacts.
        """
        raise NotImplementedError("Artifact listing not yet implemented.")

    @syncify
    @classmethod
    async def create(cls, artifact: CoreArtifact) -> Artifact:
        """
        Create a new artifact in the remote system.

        :param artifact: The core Artifact instance to create remotely.
        :return: The created Artifact instance with remote metadata.
        """
        raise NotImplementedError("Artifact creation not yet implemented.")

    @syncify
    async def delete(self) -> None:
        """
        Delete this artifact from the remote system.
        """
        raise NotImplementedError("Artifact deletion not yet implemented.")
