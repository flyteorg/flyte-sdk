from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterator, Literal, Tuple, Union

import rich.repr
from flyteidl2.common import list_pb2
from flyteidl2.project import project_service_pb2
from flyteidl2.task import run_pb2

from flyte._initialize import ensure_client, get_client
from flyte.syncify import syncify

from ._common import ToJSONMixin


@dataclass
class Project(ToJSONMixin):
    """
    A class representing a project in the Union API.
    """

    pb2: project_service_pb2.Project

    @syncify
    @classmethod
    async def get(cls, name: str) -> Project:
        """
        Get a project by name.

        :param name: The name of the project.
        :param org: The organization of the project (if applicable).
        """
        ensure_client()
        service = get_client().project_domain_service  # type: ignore
        resp = await service.get_project(
            project_service_pb2.GetProjectRequest(
                id=name,
                # org=org,
            )
        )
        return cls(resp.project)

    @syncify
    @classmethod
    async def create(
        cls,
        id: str,
        name: str,
        description: str = "",
        labels: Dict[str, str] | None = None,
    ) -> Project:
        """
        Create a new project.

        :param id: The unique identifier for the project.
        :param name: The display name for the project.
        :param description: A description for the project.
        :param labels: Optional key-value labels for the project.
        """
        ensure_client()
        project_pb = project_service_pb2.Project(
            id=id,
            name=name,
            description=description,
        )
        if labels:
            project_pb.labels.CopyFrom(run_pb2.Labels(values=labels))
        service = get_client().project_domain_service  # type: ignore
        await service.create_project(project_service_pb2.CreateProjectRequest(project=project_pb))
        return cls(project_pb)

    @syncify
    @classmethod
    async def update(
        cls,
        id: str,
        name: str | None = None,
        description: str | None = None,
        labels: Dict[str, str] | None = None,
        state: Literal["archived", "active"] | None = None,
    ) -> Project:
        """
        Update an existing project.

        :param id: The id of the project to update.
        :param name: New display name. If None, the existing name is preserved.
        :param description: New description. If None, the existing description is preserved.
        :param labels: New labels. If None, the existing labels are preserved.
        :param state: "archived" or "active". If None, the existing state is preserved.
        """
        ensure_client()
        service = get_client().project_domain_service  # type: ignore

        # Fetch current project to preserve fields not being updated
        resp = await service.get_project(project_service_pb2.GetProjectRequest(id=id))
        project_pb = resp.project
        # Clear domains — the backend rejects update requests that include them
        del project_pb.domains[:]

        if name is not None:
            project_pb.name = name
        if description is not None:
            project_pb.description = description
        if labels is not None:
            project_pb.labels.CopyFrom(run_pb2.Labels(values=labels))
        if state is not None:
            state_map = {
                "archived": project_service_pb2.PROJECT_STATE_ARCHIVED,
                "active": project_service_pb2.PROJECT_STATE_ACTIVE,
            }
            project_pb.state = state_map[state]

        await service.update_project(project_service_pb2.UpdateProjectRequest(project=project_pb))
        return cls(project_pb)

    def archive(self) -> Project:
        """Archive this project."""
        return Project.update(id=self.pb2.id, state="archived")

    def unarchive(self) -> Project:
        """Unarchive (activate) this project."""
        return Project.update(id=self.pb2.id, state="active")

    @syncify
    @classmethod
    async def listall(
        cls,
        filters: str | None = None,
        sort_by: Tuple[str, Literal["asc", "desc"]] | None = None,
        archived: bool = False,
    ) -> Union[AsyncIterator[Project], Iterator[Project]]:
        """
        List all projects.

        By default, lists active (unarchived) projects. Set `archived=True` to list
        archived projects instead.

        :param filters: The filters to apply to the project list.
        :param sort_by: The sorting criteria for the project list, in the format (field, order).
        :param archived: If True, list archived projects. If False (default), list active projects.
        :return: An iterator of projects.
        """
        ensure_client()
        token = None
        sort_by = sort_by or ("created_at", "asc")
        sort_pb2 = list_pb2.Sort(
            key=sort_by[0], direction=list_pb2.Sort.ASCENDING if sort_by[1] == "asc" else list_pb2.Sort.DESCENDING
        )

        state_value = (
            project_service_pb2.PROJECT_STATE_ARCHIVED if archived else project_service_pb2.PROJECT_STATE_ACTIVE
        )
        state_filter = f"eq(state, {state_value})"
        if filters:
            combined_filters = f"{filters}+{state_filter}"
        else:
            combined_filters = state_filter

        # org = get_common_config().org
        while True:
            resp = await get_client().project_domain_service.list_projects(  # type: ignore
                project_service_pb2.ListProjectsRequest(
                    limit=100,
                    token=token,
                    filters=combined_filters,
                    sort_by=sort_pb2,
                    # org=org,
                )
            )
            token = resp.projects.token
            for p in resp.projects.projects:
                yield cls(p)
            if not token:
                break

    def __rich_repr__(self) -> rich.repr.Result:
        yield "name", self.pb2.name
        yield "id", self.pb2.id
        yield "description", self.pb2.description
        yield "state", project_service_pb2.ProjectState.Name(self.pb2.state)
        yield (
            "labels",
            ", ".join([f"{k}: {v}" for k, v in self.pb2.labels.values.items()]) if self.pb2.labels else None,
        )
