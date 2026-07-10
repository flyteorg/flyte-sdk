from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from flyte import Link

from ._context import get_trackio_context


@dataclass
class Trackio(Link):
    """
    Generates a Trackio dashboard link for Flyte.

    The link resolution order is:

        1. Explicit server_url (self or context)
        2. Hugging Face Space (space_id)
        3. Hugging Face Trackio documentation

    Args:
        host:
            Base Hugging Face host.

        project:
            Trackio project name.

        server_url:
            Base URL of a self-hosted Trackio instance.

        space_id:
            Hugging Face Space hosting the Trackio dashboard.

        name:
            Display name in the Flyte UI.
    """

    host: str = "https://huggingface.co"

    project: Optional[str] = None

    server_url: Optional[str] = None

    space_id: Optional[str] = None

    name: str = "Trackio"

    def get_link(
        self,
        run_name: str,
        project: str,
        domain: str,
        context: Dict[str, str],
        parent_action_name: str,
        action_name: str,
        pod_name: str,
        **kwargs,
    ) -> str:
        """
        Resolve the Trackio dashboard URL.
        """

        cfg = get_trackio_context()

        project_name = self.project
        server_url = self.server_url
        space_id = self.space_id

        if cfg is not None:
            project_name = project_name or cfg.project
            server_url = server_url or cfg.server_url
            space_id = space_id or cfg.space_id

        if server_url:
            server_url = server_url.rstrip("/")

            if project_name:
                return f"{server_url}/projects/{project_name}"

            return server_url

        if space_id:
            return f"{self.host}/spaces/{space_id}"

        return f"{self.host}/docs/trackio"
