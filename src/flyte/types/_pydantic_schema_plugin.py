"""
The mashumaro JSON schema plugin for Pydantic models.

Kept out of `_type_engine` because `mashumaro.jsonschema` is expensive to import and is only
needed when the schema of a dataclass is generated. Import this module where the plugin is used.
"""

from __future__ import annotations

from mashumaro.jsonschema.models import Context, JSONSchema
from mashumaro.jsonschema.plugins import BasePlugin
from mashumaro.jsonschema.schema import Instance

from ._type_engine import CustomPydanticJsonSchemaGenerator


class PydanticSchemaPlugin(BasePlugin):
    """This allows us to generate proper schemas for Pydantic models."""

    def get_schema(
        self,
        instance: Instance,
        ctx: Context,
        schema: JSONSchema | None = None,
    ) -> JSONSchema | None:
        from pydantic import BaseModel

        try:
            if issubclass(instance.type, BaseModel):
                pydantic_schema = instance.type.model_json_schema(schema_generator=CustomPydanticJsonSchemaGenerator)
                return JSONSchema.from_dict(pydantic_schema)
        except TypeError:
            return None
        return None
