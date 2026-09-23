from dataclasses import dataclass
from typing import Dict

from flyte import Link


@dataclass
class PaperMill(Link):
    name = "PaperMill"

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

        return "https://google.com"
