from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Literal, Tuple, Union

import rich.repr
from flyteidl.admin import common_pb2, project_pb2

from flyte._initialize import ensure_client, get_client
from flyte.syncify import syncify

from ._common import ToJSONMixin


# TODO Add support for orgs again
@dataclass
class Project(ToJSONMixin):
    """
    A class representing a project in the Union API.
    """

    pb2: project_pb2.Project

    @syncify
    @classmethod
    async def get(cls, name: str, org: str | None = None) -> Project:
        """
        Get a run by its ID or name. If both are provided, the ID will take precedence.

        :param name: The name of the project.
        :param org: The organization of the project (if applicable).
        """
        ensure_client()
        service = get_client().project_domain_service  # type: ignore
        resp = await service.GetProject(
            project_pb2.ProjectGetRequest(
                id=name,
                # org=org,
            )
        )
        return cls(resp)

    @syncify
    @classmethod
    async def listall(
        cls,
        filters: str | None = None,
        sort_by: Tuple[str, Literal["asc", "desc"]] | None = None,
    ) -> Union[AsyncIterator[Project], Iterator[Project]]:
        """
        Get a run by its ID or name. If both are provided, the ID will take precedence.

        :param filters: The filters to apply to the project list.
        :param sort_by: The sorting criteria for the project list, in the format (field, order).
        :return: An iterator of projects.
        """
        ensure_client()
        token = None
        sort_by = sort_by or ("created_at", "asc")
        sort_pb2 = common_pb2.Sort(
            key=sort_by[0], direction=common_pb2.Sort.ASCENDING if sort_by[1] == "asc" else common_pb2.Sort.DESCENDING
        )
        # org = get_common_config().org
        while True:
            resp = await get_client().project_domain_service.ListProjects(  # type: ignore
                project_pb2.ProjectListRequest(
                    limit=100,
                    token=token,
                    filters=filters,
                    sort_by=sort_pb2,
                    # org=org,
                )
            )
            token = resp.token
            for p in resp.projects:
                yield cls(p)
            if not token:
                break

    def __rich_repr__(self) -> rich.repr.Result:
        yield "name", self.pb2.name
        yield "id", self.pb2.id
        yield "description", self.pb2.description
        yield "state", project_pb2.Project.ProjectState.Name(self.pb2.state)
        yield (
            "labels",
            ", ".join([f"{k}: {v}" for k, v in self.pb2.labels.values.items()]) if self.pb2.labels else None,
        )
