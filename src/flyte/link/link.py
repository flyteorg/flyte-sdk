from dataclasses import dataclass
from typing import Dict


@dataclass
class Link:
    link_type: str
    port: str

    def get_config(self) -> Dict[str, str]:
        return {
            "link_type": self.link_type,
            "port": self.port,
        }
