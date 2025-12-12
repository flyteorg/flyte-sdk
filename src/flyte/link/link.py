from typing import Dict, Protocol


class Link(Protocol):
    link_type: str
    port: str

    def get_config(
        self,
        run_name: str,
        project: str,
        domain: str,
        context: Dict[str, str],
        parent_action_name: str,
        action_name: str,
    ) -> Dict[str, str]:
        """
        Returns a dictionary representation of the link configuration.
        """
        raise NotImplementedError
