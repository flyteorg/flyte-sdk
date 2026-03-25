"""Hierarchical settings management for Flyte deployments.

Settings are scoped at two levels: DOMAIN and PROJECT. Projects inherit from
their parent domain, and domains inherit from the org-wide defaults. Narrower
scopes can override individual values.

Scope selection via ``project`` and ``domain`` parameters:

- ``domain`` only → DOMAIN scope.
- ``domain`` + ``project`` → PROJECT scope, inherits from DOMAIN.

These parameters are **independent of ``flyte.init()``**. The ``project`` and
``domain`` passed to ``init()`` configure the default execution context; they
are *not* automatically applied to settings. You must always pass ``domain``
(and optionally ``project``) explicitly.

Retrieving settings::

    import flyte.remote as remote

    # Get settings for a domain
    settings = remote.Settings.get(domain="production")

    # Get settings for a project — includes inherited DOMAIN values
    settings = remote.Settings.get(domain="production", project="ml-pipeline")

    # Inspect effective settings (resolved with inheritance)
    for s in settings.effective_settings:
        print(f"{s.key} = {s.value}  (from {s.origin})")

    # Inspect local overrides at this scope only
    for s in settings.local_settings:
        print(f"{s.key} = {s.value}")

Updating settings::

    settings = remote.Settings.get(domain="production", project="ml-pipeline")

    # Update with a dict of flat dot-notation overrides.
    # Keys not included will inherit from parent scope.
    settings.update({
        "run.default_queue": "gpu",
        "security.service_account": "ml-sa",
        "task_resource.defaults.min_cpu": 2,
    })

Interactive editing (via CLI)::

    # DOMAIN scope
    $ flyte edit settings --domain production

    # PROJECT scope
    $ flyte edit settings --domain production --project ml-pipeline

Available setting keys (dot-notation)::

    run.default_queue, run.run_concurrency, run.action_concurrency,
    security.service_account,
    storage.raw_data_path,
    task_resource.defaults.{min,max}_{cpu,gpu,memory,storage},
    task_resource.mirror_limits_request,
    labels, annotations, environment_variables
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flyteidl2.org import settings_definition_pb2, settings_service_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte.remote._common import ToJSONMixin
from flyte.syncify import syncify

# Scope level mapping from proto enum to human-readable strings
_SCOPE_NAMES = {
    settings_definition_pb2.SCOPE_LEVEL_ORG: "ORG",
    settings_definition_pb2.SCOPE_LEVEL_DOMAIN: "DOMAIN",
    settings_definition_pb2.SCOPE_LEVEL_PROJECT: "PROJECT",
}


def _extract_setting_value(sv: settings_definition_pb2.SettingValue) -> Any | None:
    """Extract the Python value from a SettingValue proto, or None if inherited/unset."""
    which = sv.WhichOneof("value")
    if which == "state":
        return None
    if which == "string_value":
        return sv.string_value
    if which == "int_value":
        return sv.int_value
    if which == "bool_value":
        return sv.bool_value
    if which == "list_value":
        return list(sv.list_value.values)
    if which == "map_value":
        return dict(sv.map_value.entries)
    return None


def _make_setting_value(value: Any) -> settings_definition_pb2.SettingValue:
    """Create a SettingValue proto from a Python value."""
    if isinstance(value, bool):
        return settings_definition_pb2.SettingValue(bool_value=value)
    elif isinstance(value, int):
        return settings_definition_pb2.SettingValue(int_value=value)
    elif isinstance(value, str):
        return settings_definition_pb2.SettingValue(string_value=value)
    elif isinstance(value, list):
        return settings_definition_pb2.SettingValue(list_value=settings_definition_pb2.StringValues(values=value))
    elif isinstance(value, dict):
        return settings_definition_pb2.SettingValue(map_value=settings_definition_pb2.StringMap(entries=value))
    else:
        return settings_definition_pb2.SettingValue(string_value=str(value))


def _settings_proto_to_flat(
    settings_proto: settings_definition_pb2.Settings,
) -> list[tuple[str, Any]]:
    """Flatten a Settings proto into dot-notation (key, value) pairs.

    Walks the nested proto structure and returns only fields that have
    actual values set (not inherited/unset).
    """
    results: list[tuple[str, Any]] = []

    def _walk(msg: Any, prefix: str) -> None:
        for fd in msg.DESCRIPTOR.fields:
            child = getattr(msg, fd.name)
            full_key = f"{prefix}.{fd.name}" if prefix else fd.name

            if fd.message_type and fd.message_type.name == "SettingValue":
                val = _extract_setting_value(child)
                if val is not None:
                    results.append((full_key, val))
            elif fd.message_type:
                # Recurse into sub-messages (RunSettings, TaskResourceSettings, etc.)
                _walk(child, full_key)

    _walk(settings_proto, "")
    return results


# Map of dot-notation prefixes to their proto sub-message class and parent field name.
# This drives the reverse conversion from flat keys to nested proto.
_SETTINGS_TREE: dict[str, tuple[str, type]] = {
    "run": ("run", settings_definition_pb2.RunSettings),
    "security": ("security", settings_definition_pb2.SecuritySettings),
    "storage": ("storage", settings_definition_pb2.StorageSettings),
    "task_resource": ("task_resource", settings_definition_pb2.TaskResourceSettings),
    "task_resource.defaults": ("defaults", settings_definition_pb2.TaskResourceDefaults),
}


def _flat_overrides_to_settings_proto(
    overrides: dict[str, Any],
) -> settings_definition_pb2.Settings:
    """Build a Settings proto from flat dot-notation overrides.

    Example input: {"run.default_queue": "gpu", "task_resource.defaults.min_cpu": 2}
    """
    settings = settings_definition_pb2.Settings()

    for dotkey, value in overrides.items():
        sv = _make_setting_value(value)
        parts = dotkey.split(".")

        # Top-level SettingValue fields (labels, annotations, environment_variables)
        if len(parts) == 1:
            getattr(settings, parts[0]).CopyFrom(sv)
            continue

        # Two-part keys: e.g., run.default_queue, security.service_account
        if len(parts) == 2:
            parent = getattr(settings, parts[0])
            getattr(parent, parts[1]).CopyFrom(sv)
            continue

        # Three-part keys: e.g., task_resource.defaults.min_cpu
        if len(parts) == 3:
            mid = getattr(settings, parts[0])
            parent = getattr(mid, parts[1])
            getattr(parent, parts[2]).CopyFrom(sv)
            continue

    return settings


def _scope_origin_from_key(
    key: settings_definition_pb2.SettingsKey,
) -> SettingOrigin:
    """Determine the SettingOrigin from a SettingsKey proto."""
    if key.project:
        return SettingOrigin(scope_type="PROJECT", domain=key.domain, project=key.project)
    elif key.domain:
        return SettingOrigin(scope_type="DOMAIN", domain=key.domain)
    else:
        return SettingOrigin(scope_type="ORG")


@dataclass
class SettingOrigin:
    """Represents where a setting value comes from in the hierarchy."""

    scope_type: str  # "ORG", "DOMAIN", or "PROJECT"
    domain: str | None = None
    project: str | None = None

    def __str__(self) -> str:
        if self.scope_type == "ORG":
            return "ORG"
        elif self.scope_type == "DOMAIN":
            return f"DOMAIN({self.domain})"
        else:
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

    This class manages settings across ORG, DOMAIN, and PROJECT scopes,
    supporting local overrides and inheritance from parent scopes.
    """

    effective_settings: list[EffectiveSetting]
    local_settings: list[LocalSetting]
    domain: str | None = None
    project: str | None = None
    _version: int = field(default=0, repr=False)

    @staticmethod
    def _parse_value(raw_value: str) -> Any:
        """Parse a YAML value string into appropriate Python type."""
        raw_value = raw_value.strip()

        if (raw_value.startswith('"') and raw_value.endswith('"')) or (
            raw_value.startswith("'") and raw_value.endswith("'")
        ):
            return raw_value[1:-1]

        if raw_value.lower() == "true":
            return True
        if raw_value.lower() == "false":
            return False

        try:
            if "." in raw_value:
                return float(raw_value)
            return int(raw_value)
        except ValueError:
            pass

        return raw_value

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a Python value for YAML output."""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
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
        local_keys = {s.key for s in self.local_settings}

        if self.local_settings:
            lines.append("# Local overrides")
            for l_setting in self.local_settings:
                value_str = self._format_value(l_setting.value)
                lines.append(f"{l_setting.key}: {value_str}")
            lines.append("")

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
            if not stripped or stripped.startswith("#"):
                continue

            if ":" not in stripped:
                continue

            key, raw_value = stripped.split(":", 1)
            if "#" in raw_value:
                raw_value = raw_value.split("#")[0]

            overrides[key.strip()] = Settings._parse_value(raw_value)

        return overrides

    @syncify
    @classmethod
    async def get(cls, project: str | None = None, domain: str | None = None) -> Settings:
        """Retrieve settings for a given scope.

        Returns a Settings object containing both the effective (resolved) settings
        with inheritance, and the local overrides at the requested scope.

        The scope is determined by ``domain`` and ``project``:

        - ``domain`` only → DOMAIN scope.
        - ``domain`` + ``project`` → PROJECT scope, inherits from DOMAIN.

        These are explicit parameters — they are **not** inferred from
        ``flyte.init()``.

        :param domain: Domain name (required).
        :param project: Project name. Requires ``domain`` to also be set.
        :returns: Settings object with effective_settings, local_settings, and version.

        Example::

            # Domain-level settings
            settings = Settings.get(domain="production")

            # Project-level — inherits from DOMAIN
            settings = Settings.get(domain="production", project="ml-pipeline")
        """
        ensure_client()
        cfg = get_init_config()
        client = get_client()

        key = settings_definition_pb2.SettingsKey(org=cfg.org or "", domain=domain or "", project=project or "")
        resp = await client.settings_service.GetSettingsForEdit(settings_service_pb2.GetSettingsForEditRequest(key=key))

        # resp.levels is ordered broadest → most specific.
        # Build effective settings by layering: broader values get overridden by narrower.
        effective: dict[str, EffectiveSetting] = {}
        for record in resp.levels:
            origin = _scope_origin_from_key(record.key)
            for dotkey, val in _settings_proto_to_flat(record.settings):
                effective[dotkey] = EffectiveSetting(key=dotkey, value=val, origin=origin)

        # Local settings come from the most specific level that matches the requested scope.
        local_settings: list[LocalSetting] = []
        version = 0
        if resp.levels:
            most_specific = resp.levels[-1]
            version = most_specific.version
            for dotkey, val in _settings_proto_to_flat(most_specific.settings):
                local_settings.append(LocalSetting(key=dotkey, value=val))

        return cls(
            effective_settings=list(effective.values()),
            local_settings=local_settings,
            domain=domain,
            project=project,
            _version=version,
        )

    @syncify
    async def update(self, overrides: dict[str, Any]) -> None:
        """Update settings with new local overrides at this scope.

        Replaces the complete set of local overrides for the scope that this
        Settings object was retrieved for (determined by the ``domain`` and
        ``project`` passed to ``get()``). Settings not included in ``overrides``
        will inherit from the parent scope.

        Uses optimistic locking via the version obtained from ``get()``.

        :param overrides: Dict of flat dot-notation keys to values.
            Example: ``{"run.default_queue": "gpu", "security.service_account": "my-sa"}``

        Example::

            settings = Settings.get(domain="production")
            settings.update({
                "run.default_queue": "gpu",
                "task_resource.defaults.min_cpu": 2,
            })
        """
        cfg = get_init_config()
        client = get_client()

        key = settings_definition_pb2.SettingsKey(
            org=cfg.org or "", domain=self.domain or "", project=self.project or ""
        )
        settings_proto = _flat_overrides_to_settings_proto(overrides)

        await client.settings_service.UpdateSettings(
            settings_service_pb2.UpdateSettingsRequest(key=key, settings=settings_proto, version=self._version)
        )

        # Update local state
        self.local_settings = [LocalSetting(key=k, value=v) for k, v in overrides.items()]
