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
        :param run_name: The name of the run.
        :param project: The project name.
        :param domain: The domain name.
        :param context: Additional context for generating the link.
        :param parent_action_name: The name of the parent action.
        :param action_name: The name of the action.
        :param pod_name: The name of the pod.
        :param kwargs: Additional keyword arguments.
        :return: The generated link.
        """
        raise NotImplementedError
