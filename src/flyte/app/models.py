from dataclasses import dataclass, field
from typing import Optional

from flyte._protos.app.app_definition_pb2 import Spec
from flyte.app._app_environment_bk import MaterializedInput


@dataclass
class AppSerializationSettings:
    """Runtime settings for creating an AppIDL"""

    org: str
    project: str
    domain: str
    version: str
    image_uri: str
    is_serverless: bool
    desired_state: Spec.DesiredState
    materialized_inputs: dict[str, MaterializedInput] = field(default_factory=dict)
    additional_distribution: Optional[str] = None
