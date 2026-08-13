from typing import Dict, Optional, Protocol


class Link(Protocol):
    name: str
    icon_uri: Optional[str] = ""

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
        Returns a task log link given the action.
        Link can have template variables that are replaced by the backend.

        Args:
            run_name: The name of the run.
            project: The project name.
            domain: The domain name.
            context: Additional context for generating the link.
            parent_action_name: The name of the parent action.
            action_name: The name of the action.
            pod_name: The name of the pod.
            kwargs: Additional keyword arguments.

        Returns:
            The generated link.
        """
        raise NotImplementedError
