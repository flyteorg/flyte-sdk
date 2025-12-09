from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Link:
    link_type: str
    port: str

    def get_config(self) -> Dict[str, str]:
        return {
            "link_type": self.link_type,
            "port": self.port,
        }
