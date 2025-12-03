from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flyte.remote._common import ToJSONMixin
from flyte.remote._settings_store import SettingsStore
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
    _store: SettingsStore | None = None

    def __post_init__(self):
        """Initialize the settings store if not provided."""
        if self._store is None:
            object.__setattr__(self, "_store", SettingsStore())

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
            for l_setting in self.local_settings:
                value_str = self._format_value(l_setting.value)
                lines.append(f"{l_setting.key}: {value_str}")
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
    async def get(
        cls, project: str | None = None, domain: str | None = None, store_path: Path | None = None
    ) -> Settings:
        """Retrieve settings for a given scope.

        Args:
            project: Project name (requires domain to be set)
            domain: Domain name
            store_path: Optional custom path to settings store file

        Returns:
            Settings object with effective and local settings
        """
        # TODO: Once the gRPC client is implemented, replace local store with API call
        # from flyte.remote._client.controlplane import ClientSet
        # client = await ClientSet.from_env()
        # request = GetSettingsRequest(domain=domain or "", project=project or "")
        # response = await client.settings_service.GetSettings(request)

        # Use local file-based store
        store = SettingsStore(store_path)

        # Get effective settings with inheritance
        effective_dict = store.get_effective_settings(domain=domain, project=project)
        effective_settings = [
            EffectiveSetting(
                key=key,
                value=value,
                origin=SettingOrigin(scope_type=scope_type, domain=origin_domain, project=origin_project),
            )
            for key, (value, scope_type, origin_domain, origin_project) in effective_dict.items()
        ]

        # Get local settings for this scope only
        local_dict = store.get_local_settings(domain=domain, project=project)
        local_settings = [LocalSetting(key=k, value=v) for k, v in local_dict.items()]

        return cls(
            effective_settings=effective_settings,
            local_settings=local_settings,
            domain=domain,
            project=project,
            _store=store,
        )

    @syncify
    async def update(self, overrides: dict[str, Any]) -> None:
        """Update settings with new local overrides.

        This replaces the complete set of local overrides for the current scope.
        Settings not included will be removed and inherit from parent scope.

        Args:
            overrides: Dictionary of setting key-value pairs to set as local overrides
        """
        # TODO: Once the gRPC client is implemented, replace local store with API call
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

        # Update local file-based store
        if self._store is None:
            self._store = SettingsStore()

        self._store.set_local_settings(overrides, domain=self.domain, project=self.project)

        # Update local state to reflect the changes
        self.local_settings = [LocalSetting(key=k, value=v) for k, v in overrides.items()]

        # Recalculate effective settings with inheritance
        effective_dict = self._store.get_effective_settings(domain=self.domain, project=self.project)
        self.effective_settings = [
            EffectiveSetting(
                key=key,
                value=value,
                origin=SettingOrigin(scope_type=scope_type, domain=origin_domain, project=origin_project),
            )
            for key, (value, scope_type, origin_domain, origin_project) in effective_dict.items()
        ]
