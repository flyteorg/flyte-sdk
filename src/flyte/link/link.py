from typing import Dict, Protocol


class Link(Protocol):
    link_type: str
    port: str

    def get_config(self) -> Dict[str, str]:
        """
        Returns a dictionary representation of the link configuration.
        """
        raise NotImplementedError
