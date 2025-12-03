from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flyte.remote._common import ToJSONMixin
from flyte.syncify import syncify


@dataclass
class SettingOrigin:
    """Represents where a setting value comes from in the hierarchy."""

    scope_type: str  # "ROOT", "DOMAIN", or "PROJECT"
    domain: str | None = None
    project: str | None = None

    def __str__(self) -> str:
        if self.scope_type == "ROOT":
            return "ROOT"
        elif self.scope_type == "DOMAIN":
            return f"DOMAIN({self.domain})"
        else:  # PROJECT
            return f"PROJECT({self.domain}/{self.project})"


@dataclass
class EffectiveSetting:
    """Represents a resolved setting value with its origin."""

    key: str
    value: Any
    origin: SettingOrigin


@dataclass
class LocalSetting:
    """Represents an explicit override at a specific scope."""

    key: str
    value: Any


@dataclass
class Settings(ToJSONMixin):
    """Hierarchical configuration settings with inheritance support.

    This class manages settings across ROOT, DOMAIN, and PROJECT scopes,
    supporting local overrides and inheritance from parent scopes.
    """

    effective_settings: list[EffectiveSetting]
    local_settings: list[LocalSetting]
    domain: str | None = None
    project: str | None = None

    @staticmethod
    def _parse_value(raw_value: str) -> Any:
        """Parse a YAML value string into appropriate Python type."""
        raw_value = raw_value.strip()

        # Remove quotes if present
        if (raw_value.startswith('"') and raw_value.endswith('"')) or (
            raw_value.startswith("'") and raw_value.endswith("'")
        ):
            return raw_value[1:-1]

        # Parse booleans
        if raw_value.lower() == "true":
            return True
        if raw_value.lower() == "false":
            return False

        # Parse numbers
        try:
            if "." in raw_value:
                return float(raw_value)
            return int(raw_value)
        except ValueError:
            pass

        # Return as string
        return raw_value

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a Python value for YAML output."""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            # Quote strings that contain special characters or are numeric-looking
            if (
                any(c in value for c in [":", "#", "\n", '"', "'"])
                or value.strip() != value
                or (value and (value[0].isdigit() or value in ["true", "false", "null"]))
            ):
                return f'"{value}"'
            return value
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return str(value)

    def to_yaml(self) -> str:
        """Generate YAML representation showing local and inherited settings.

        Local settings are uncommented, inherited settings are commented with origin info.
        """
        lines = []

        # Create a set of local setting keys for quick lookup
        local_keys = {s.key for s in self.local_settings}

        # Add local overrides section
        if self.local_settings:
            lines.append("# Local overrides")
            for setting in self.local_settings:
                value_str = self._format_value(setting.value)
                lines.append(f"{setting.key}: {value_str}")
            lines.append("")

        # Add inherited settings section
        inherited = [s for s in self.effective_settings if s.key not in local_keys]
        if inherited:
            lines.append("# Inherited settings (uncomment to override)")
            for setting in inherited:
                value_str = self._format_value(setting.value)
                lines.append(f"# {setting.key}: {value_str}  # inherited from {setting.origin}")

        return "\n".join(lines)

    @staticmethod
    def parse_yaml(yaml_content: str) -> dict[str, Any]:
        """Parse edited YAML content into a dictionary of overrides.

        Only uncommented lines are considered as local overrides.
        Commented lines are treated as removed/inherited.
        """
        overrides = {}
        for line in yaml_content.split("\n"):
            stripped = line.strip()
            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                continue

            # Parse key-value pair
            if ":" not in stripped:
                continue

            key, raw_value = stripped.split(":", 1)
            # Remove inline comments
            if "#" in raw_value:
                raw_value = raw_value.split("#")[0]

            overrides[key.strip()] = Settings._parse_value(raw_value)

        return overrides

    @syncify
    @classmethod
    async def get(cls, project: str | None = None, domain: str | None = None) -> Settings:
        """Retrieve settings for a given scope.

        Args:
            project: Project name (requires domain to be set)
            domain: Domain name

        Returns:
            Settings object with effective and local settings
        """
        # TODO: Once the gRPC client is implemented, replace this with actual API call
        # from flyte.remote._client.controlplane import ClientSet
        # client = await ClientSet.from_env()
        # request = GetSettingsRequest(domain=domain or "", project=project or "")
        # response = await client.settings_service.GetSettings(request)

        # For now, return dummy data
        effective_settings = [
            EffectiveSetting(key="default_queue", value="default", origin=SettingOrigin(scope_type="ROOT")),
            EffectiveSetting(key="run_concurrency", value=10, origin=SettingOrigin(scope_type="ROOT")),
            EffectiveSetting(key="interruptible", value=False, origin=SettingOrigin(scope_type="ROOT")),
        ]

        local_settings = []
        if domain:
            # Add some domain-level overrides for demo
            local_settings.append(LocalSetting(key="run_concurrency", value=20))
            effective_settings[1] = EffectiveSetting(
                key="run_concurrency", value=20, origin=SettingOrigin(scope_type="DOMAIN", domain=domain)
            )

        return cls(effective_settings=effective_settings, local_settings=local_settings, domain=domain, project=project)

    @syncify
    async def update(self, overrides: dict[str, Any]) -> None:
        """Update settings with new local overrides.

        This replaces the complete set of local overrides for the current scope.
        Settings not included will be removed and inherit from parent scope.

        Args:
            overrides: Dictionary of setting key-value pairs to set as local overrides
        """
        # TODO: Once the gRPC client is implemented, replace this with actual API call
        # from flyte.remote._client.controlplane import ClientSet
        # client = await ClientSet.from_env()
        #
        # local_settings = [
        #     LocalSetting(key=k, value=v) for k, v in overrides.items()
        # ]
        #
        # request = UpdateSettingsRequest(
        #     domain=self.domain or "",
        #     project=self.project or "",
        #     local_settings=local_settings
        # )
        # await client.settings_service.UpdateSettings(request)

        # For now, just update the local state
        self.local_settings = [LocalSetting(key=k, value=v) for k, v in overrides.items()]

        # Update effective settings to reflect the changes
        effective_dict = {s.key: s for s in self.effective_settings}
        for key, value in overrides.items():
            origin = SettingOrigin(
                scope_type="PROJECT" if self.project else ("DOMAIN" if self.domain else "ROOT"),
                domain=self.domain,
                project=self.project,
            )
            effective_dict[key] = EffectiveSetting(key=key, value=value, origin=origin)

        self.effective_settings = list(effective_dict.values())
