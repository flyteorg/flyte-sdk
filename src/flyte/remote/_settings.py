"""Hierarchical settings management for Flyte deployments.

Settings are scoped at three levels: ORG, DOMAIN, and PROJECT. Projects inherit
from their parent domain, and domains inherit from the org-wide defaults. Narrower
scopes can override individual values.

Scope selection via ``project`` and ``domain`` parameters:

- no args → ORG scope.
- ``domain`` only → DOMAIN scope, inherits from ORG.
- ``domain`` + ``project`` → PROJECT scope, inherits from DOMAIN.

These parameters are **independent of ``flyte.init()``**. The ``project`` and
``domain`` passed to ``init()`` configure the default execution context; they
are *not* automatically applied to settings. You must always pass ``domain``
(and optionally ``project``) explicitly.

Retrieving settings::

    import flyte.remote as remote

    # Get settings for a domain
    settings = remote.Settings.get_settings_for_edit(domain="production")

    # Get settings for a project — includes inherited DOMAIN values
    settings = remote.Settings.get_settings_for_edit(domain="production", project="ml-pipeline")

    # Inspect effective settings (resolved with inheritance)
    for s in settings.effective_settings:
        print(f"{s.key} = {s.value}  (from {s.origin})")

    # Inspect local overrides at this scope only
    for s in settings.local_settings:
        print(f"{s.key} = {s.value}")

Updating settings::

    settings = remote.Settings.get_settings_for_edit(domain="production", project="ml-pipeline")

    # Update with a dict of flat dot-notation overrides.
    # Keys not included will inherit from parent scope.
    settings.update_settings({
        "run.default_queue": "gpu",
        "security.service_account": "ml-sa",
        "task_resource.min.cpu": "2",
    })

Interactive editing (via CLI)::

    # DOMAIN scope
    $ flyte edit settings --domain production

    # PROJECT scope
    $ flyte edit settings --domain production --project ml-pipeline

Available setting keys (dot-notation)::

    run.default_queue,
    security.service_account,
    storage.raw_data_path,
    task_resource.min.{cpu,gpu,memory,storage},
    task_resource.max.{cpu,gpu,memory,storage},
    task_resource.mirror_limits_request,
    labels, annotations, environment_variables
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from flyteidl2.settings import settings_definition_pb2, settings_service_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte.remote._common import ToJSONMixin
from flyte.syncify import syncify


class _UnsetType:
    """Sentinel for SETTING_STATE_UNSET — explicitly clears a setting, blocking parent inheritance."""

    _instance = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _UnsetType()
"""Pass this value to explicitly unset a setting at the current scope.

An UNSET setting blocks inheritance from the parent scope; the setting has no
value at this scope even if a parent scope defines one. Distinct from simply
omitting the key (which means "inherit").

In YAML, write ``key: ~unset`` to express this state.
"""

_YAML_UNSET_TOKEN = "~unset"
_MAP_MERGE_COMMENT = (
    "## Map entries merge across scopes — parent entries are shown below and applied unless overridden here."
)
_LIST_ADDITIVE_COMMENT = (
    "## List values add across scopes — parent items are shown below and included unless overridden here."
)

# Scope level mapping from proto enum to human-readable strings
_SCOPE_NAMES = {
    settings_definition_pb2.SCOPE_LEVEL_ORG: "ORG",
    settings_definition_pb2.SCOPE_LEVEL_DOMAIN: "DOMAIN",
    settings_definition_pb2.SCOPE_LEVEL_PROJECT: "PROJECT",
}

# Schema of all settable leaves in the Settings proto, derived at import time
# by walking ``settings_definition_pb2.Settings.DESCRIPTOR``. Each entry is
# ``(dotkey, leaf_kind, field_descriptor)``.
#
# Leaf kinds map to the proto *Setting types:
#     "string"     → StringSetting
#     "int"        → Int64Setting
#     "bool"       → BoolSetting
#     "quantity"   → QuantitySetting (string-encoded k8s quantity)
#     "stringlist" → StringListSetting
#     "stringmap"  → StringMapSetting


@dataclass(frozen=True)
class _LeafIO:
    """Per-kind I/O for ``*Setting`` leaf messages — single source of truth for
    leaf marshalling. ``extract`` reads a Python value out of a populated leaf;
    ``build_kwargs`` returns the proto-class kwargs (excluding ``state``) for a
    Python value.
    """

    proto_class: type
    extract: Callable[[Any], Any]
    build_kwargs: Callable[[Any], dict[str, Any]]


def _build_list_kwargs(v: Any) -> dict[str, Any]:
    if not isinstance(v, (list, tuple)):
        raise TypeError(f"expected list for stringlist leaf, got {type(v).__name__}")
    return {"list_value": settings_definition_pb2.StringValues(values=[str(x) for x in v])}


def _build_map_kwargs(v: Any) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise TypeError(f"expected dict for stringmap leaf, got {type(v).__name__}")
    return {"map_value": settings_definition_pb2.StringMap(entries={str(k): str(val) for k, val in v.items()})}


_LEAF_IO: dict[str, _LeafIO] = {
    "string": _LeafIO(
        settings_definition_pb2.StringSetting,
        lambda leaf: leaf.string_value,
        lambda v: {"string_value": str(v)},
    ),
    "int": _LeafIO(
        settings_definition_pb2.Int64Setting,
        lambda leaf: leaf.int_value,
        lambda v: {"int_value": int(v)},
    ),
    "bool": _LeafIO(
        settings_definition_pb2.BoolSetting,
        lambda leaf: leaf.bool_value,
        lambda v: {"bool_value": bool(v)},
    ),
    "quantity": _LeafIO(
        settings_definition_pb2.QuantitySetting,
        lambda leaf: leaf.quantity_value,
        lambda v: {"quantity_value": str(v)},
    ),
    "stringlist": _LeafIO(
        settings_definition_pb2.StringListSetting,
        lambda leaf: list(leaf.list_value.values),
        _build_list_kwargs,
    ),
    "stringmap": _LeafIO(
        settings_definition_pb2.StringMapSetting,
        lambda leaf: dict(leaf.map_value.entries),
        _build_map_kwargs,
    ),
}

# Lookup from proto class name → kind, used by ``_discover_leaves`` to identify
# which descriptor fields are leaves.
_SETTING_TYPE_TO_KIND: dict[str, str] = {io.proto_class.__name__: kind for kind, io in _LEAF_IO.items()}

# YAML-formatted placeholder per leaf kind, used as the default value in the
# "Available settings" template section. The proto's ``desc`` extension carries
# the concrete format hints (e.g. ``'256Mi'`` for memory quantities).
_PLACEHOLDER_BY_KIND: dict[str, str] = {
    "string": "''",
    "int": "0",
    "bool": "false",
    "quantity": "''",
    "stringlist": "[]",
    "stringmap": "{}",
}


def _resolve_description(field_descriptor: Any) -> str:
    """Return the ``desc`` field option extension attached to a proto field."""
    options = field_descriptor.GetOptions()
    ext = settings_definition_pb2.desc
    if options.HasExtension(ext):
        return options.Extensions[ext]
    return ""


def _discover_leaves(descriptor: Any, prefix: str = "") -> list[tuple[str, str, Any]]:
    """Recursively walk a ``Settings`` message descriptor, emitting one tuple
    per leaf ``*Setting`` field encountered."""
    results: list[tuple[str, str, Any]] = []
    for fd in descriptor.fields:
        if fd.type != fd.TYPE_MESSAGE or fd.message_type is None:
            continue
        dotkey = f"{prefix}.{fd.name}" if prefix else fd.name
        kind = _SETTING_TYPE_TO_KIND.get(fd.message_type.name)
        if kind is not None:
            results.append((dotkey, kind, fd))
        else:
            results.extend(_discover_leaves(fd.message_type, dotkey))
    return results


_LEAF_SCHEMA: list[tuple[str, str, Any]] = _discover_leaves(settings_definition_pb2.Settings.DESCRIPTOR)
_LEAF_TYPES: dict[str, str] = {dotkey: kind for dotkey, kind, _ in _LEAF_SCHEMA}
_LEAF_DESCRIPTIONS: dict[str, str] = {dotkey: _resolve_description(fd) for dotkey, _, fd in _LEAF_SCHEMA}
_LEAF_EXAMPLES: dict[str, str] = {dotkey: _PLACEHOLDER_BY_KIND[kind] for dotkey, kind, _ in _LEAF_SCHEMA}


def _extract_leaf_value(leaf: Any, leaf_type: str) -> Any | None:
    """Extract the Python value from a typed ``*Setting`` proto.

    Returns ``None`` for INHERIT state, :data:`UNSET` for UNSET state, or the
    concrete value for VALUE state.
    """
    if leaf.state == settings_definition_pb2.SETTING_STATE_UNSET:
        return UNSET
    if leaf.state != settings_definition_pb2.SETTING_STATE_VALUE:
        return None
    io = _LEAF_IO.get(leaf_type)
    return io.extract(leaf) if io else None


def _build_leaf(leaf_type: str, value: Any) -> Any:
    """Build a typed ``*Setting`` proto from a Python value.

    When ``value`` is :data:`UNSET`, the leaf carries ``state=SETTING_STATE_UNSET``
    and no payload fields. Otherwise ``state=SETTING_STATE_VALUE``.
    """
    io = _LEAF_IO.get(leaf_type)
    if io is None:
        raise ValueError(f"unknown leaf type: {leaf_type}")
    if value is UNSET:
        return io.proto_class(state=settings_definition_pb2.SETTING_STATE_UNSET)
    return io.proto_class(state=settings_definition_pb2.SETTING_STATE_VALUE, **io.build_kwargs(value))


def _walk_leaf(settings_msg: settings_definition_pb2.Settings, dotkey: str) -> Any | None:
    """Navigate dot-notation ``dotkey`` into a Settings proto. Returns the leaf
    ``*Setting`` message if every intermediate submessage is present, else ``None``.
    """
    obj = settings_msg
    for part in dotkey.split("."):
        if not obj.HasField(part):
            return None
        obj = getattr(obj, part)
    return obj


def _proto_to_flat(settings_msg: settings_definition_pb2.Settings) -> list[tuple[str, Any]]:
    """Extract all leaves with an explicit VALUE state as flat (dotkey, value) pairs."""
    results: list[tuple[str, Any]] = []
    for dotkey, leaf_type, _ in _LEAF_SCHEMA:
        leaf = _walk_leaf(settings_msg, dotkey)
        if leaf is None:
            continue
        val = _extract_leaf_value(leaf, leaf_type)
        if val is not None:
            results.append((dotkey, val))
    return results


def _flat_to_proto(overrides: dict[str, Any]) -> settings_definition_pb2.Settings:
    """Build a Settings proto from flat dot-notation overrides.

    Keys not present in ``overrides`` are left unset in the returned proto,
    which the server interprets as "inherit from parent scope".
    """
    msg = settings_definition_pb2.Settings()
    for dotkey, value in overrides.items():
        if dotkey not in _LEAF_TYPES:
            raise ValueError(f"unknown settings key: {dotkey!r}. valid keys: {', '.join(sorted(_LEAF_TYPES))}")
        parts = dotkey.split(".")
        parent = msg
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf_proto = _build_leaf(_LEAF_TYPES[dotkey], value)
        getattr(parent, parts[-1]).CopyFrom(leaf_proto)
    return msg


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
    _parent_effective: dict[str, EffectiveSetting] = field(default_factory=dict, repr=False)
    _map_entry_origins: dict[str, dict[str, EffectiveSetting]] = field(default_factory=dict, repr=False)
    _list_item_origins: dict[str, list[EffectiveSetting]] = field(default_factory=dict, repr=False)

    @staticmethod
    def available_keys() -> list[str]:
        """Return every dot-notation key that can be set on a Settings scope."""
        return [k for k, _, _ in _LEAF_SCHEMA]

    def local_overrides(self) -> dict[str, Any]:
        """Return local overrides as a flat dict for programmatic use."""
        return {s.key: s.value for s in self.local_settings}

    def effective_values(self) -> dict[str, Any]:
        """Return resolved effective settings as a flat dict for programmatic use."""
        return {s.key: s.value for s in self.effective_settings}

    def scope_description(self) -> str:
        """Human-readable label for the scope this Settings object was fetched for."""
        if self.project and self.domain:
            return f"PROJECT({self.domain}/{self.project})"
        if self.domain:
            return f"DOMAIN({self.domain})"
        return "ORG"

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a Python value as a YAML scalar / flow collection."""
        if value is UNSET:
            return _YAML_UNSET_TOKEN
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            if (
                any(c in value for c in [":", "#", "\n", '"', "'"])
                or value.strip() != value
                or (value and (value[0].isdigit() or value in ["true", "false", "null"]))
            ):
                return f'"{value}"'
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (list, tuple, dict)):
            import yaml

            return yaml.safe_dump(value, default_flow_style=True, width=10000).strip()
        return str(value)

    def _render_merge_collection(self, l_setting: LocalSetting) -> list[str]:
        """Render a stringmap or stringlist local override as YAML lines, with
        parent-scope items shown as ``#``-commented lines tagged with their
        origin scope.

        When the local container has entries::

            ## merge/additive comment
            {key}:
              {local entries...}
              # {parent entry}  ## defined at {origin}

        When the local container is empty, the key line itself is commented so
        saving an unedited file does not replace inherited values with ``null``::

            ## merge/additive comment
            # {key}:
            #   {parent entry}  ## defined at {origin}
        """
        key = l_setting.key
        if _LEAF_TYPES.get(key) == "stringmap":
            merge_comment = _MAP_MERGE_COMMENT
            local_dict = l_setting.value if isinstance(l_setting.value, dict) else {}
            local_items = [(k, f"{k}: {self._format_value(v)}") for k, v in local_dict.items()]
            parent_items = [
                (k, f"{k}: {self._format_value(eff.value)}", eff.origin)
                for k, eff in self._map_entry_origins.get(key, {}).items()
            ]
        else:  # stringlist
            merge_comment = _LIST_ADDITIVE_COMMENT
            local_list = l_setting.value if isinstance(l_setting.value, list) else []
            local_items = [(v, f"- {self._format_value(v)}") for v in local_list]
            parent_items = [
                (eff.value, f"- {self._format_value(eff.value)}", eff.origin)
                for eff in self._list_item_origins.get(key, [])
            ]

        local_dedup_keys = {dk for dk, _ in local_items}
        lines = [merge_comment]
        if local_items:
            lines.append(f"{key}:")
            lines.extend(f"  {repr_str}" for _, repr_str in local_items)
            for dk, repr_str, origin in parent_items:
                if dk not in local_dedup_keys:
                    lines.append(f"  # {repr_str}  ## defined at {origin}")
        else:
            lines.append(f"# {key}:")
            for _, repr_str, origin in parent_items:
                lines.append(f"#   {repr_str}  ## defined at {origin}")
        return lines

    def to_yaml_sections(self) -> list[tuple[str, str]]:
        """Return the YAML content split into labelled sections.

        Each tuple is ``(section_title, yaml_body)``; sections are omitted
        when they have no entries. See :meth:`to_yaml` for the comment-prefix
        convention.

        Section titles: ``"Local overrides"``, ``"Inherited settings"``, ``"Available settings"``.
        """
        local_keys = {s.key for s in self.local_settings}
        effective_keys = {s.key for s in self.effective_settings}

        # A stringmap or stringlist local setting with no local entries has nothing to override
        # — treat it as inherited for display so it appears in the Inherited section, not Local overrides.
        empty_local_map_keys = {
            s.key
            for s in self.local_settings
            if (
                (_LEAF_TYPES.get(s.key) == "stringmap" and not (isinstance(s.value, dict) and s.value))
                or (_LEAF_TYPES.get(s.key) == "stringlist" and not (isinstance(s.value, list) and s.value))
            )
        }
        display_local_settings = [s for s in self.local_settings if s.key not in empty_local_map_keys]
        display_local_keys = local_keys - empty_local_map_keys

        def _describe(dotkey: str) -> str | None:
            desc = _LEAF_DESCRIPTIONS.get(dotkey)
            return f"## {desc}" if desc else None

        sections: list[tuple[str, str]] = []

        if display_local_settings:
            lines: list[str] = []
            for l_setting in display_local_settings:
                d = _describe(l_setting.key)
                if d:
                    lines.append(d)
                if _LEAF_TYPES.get(l_setting.key) in ("stringmap", "stringlist"):
                    lines.extend(self._render_merge_collection(l_setting))
                else:
                    lines.append(f"{l_setting.key}: {self._format_value(l_setting.value)}")
                    parent = self._parent_effective.get(l_setting.key)
                    if parent is not None:
                        lines.append(
                            f"# overridden value: {self._format_value(parent.value)}  ## defined at {parent.origin}"
                        )
            sections.append(("Local overrides", "\n".join(lines)))

        inherited = [s for s in self.effective_settings if s.key not in display_local_keys]
        if inherited:
            lines = []
            for setting in inherited:
                d = _describe(setting.key)
                if d:
                    lines.append(d)
                if _LEAF_TYPES.get(setting.key) == "stringmap" and isinstance(setting.value, dict):
                    lines.append(f"# {setting.key}:  ## inherited from {setting.origin}")
                    for entry_key, entry_val in setting.value.items():
                        lines.append(f"#   {entry_key}: {self._format_value(entry_val)}")
                elif _LEAF_TYPES.get(setting.key) == "stringlist" and isinstance(setting.value, list):
                    lines.append(f"# {setting.key}:  ## inherited from {setting.origin}")
                    for item in setting.value:
                        lines.append(f"#   - {self._format_value(item)}")
                else:
                    lines.append(
                        f"# {setting.key}: {self._format_value(setting.value)}  ## inherited from {setting.origin}"
                    )
            sections.append(("Inherited settings", "\n".join(lines)))

        unset = [k for k, _, _ in _LEAF_SCHEMA if k not in display_local_keys and k not in effective_keys]
        if unset:
            lines = []
            for key in unset:
                d = _describe(key)
                if d:
                    lines.append(d)
                lines.append(f"# {key}: {_LEAF_EXAMPLES[key]}")
            sections.append(("Available settings", "\n".join(lines)))

        return sections

    def to_yaml(self) -> str:
        """Generate YAML representation of this scope.

        Three comment prefixes form a visibility hierarchy for both human
        readers and line-based stylers:

        * ``###`` — section headers (scope header, section titles). Prominent.
        * ``##`` — descriptions and metadata (per-field docs, inline origin
          annotations). Dim.
        * ``#`` — a commented-out setting. Uncomment (strip a single leading
          ``#``) to activate.

        A bulk ``# `` → `` pass safely activates every setting while leaving
        descriptions (``##`` → ``#``) and section headers (``###`` → ``##``)
        intact as comments.

        Output has up to three sections:

        * **Local overrides** — already applied at this scope.
        * **Inherited settings** — resolved from a parent scope; uncomment to
          override here.
        * **Available settings** — every key not yet set anywhere, with
          illustrative placeholders; uncomment and edit to set it here.
        """
        _SECTION_HEADERS = {
            "Local overrides": "### Local overrides",
            "Inherited settings": "### Inherited settings (uncomment to override at this scope)",
            "Available settings": "### Available settings (uncomment and edit to set at this scope)",
        }

        lines = [
            f"### Settings for scope: {self.scope_description()}",
            "## Remove or comment out a line to inherit that setting from the parent scope.",
            "## Set a value to ~unset to explicitly clear it, blocking parent inheritance.",
            "",
        ]
        for title, body in self.to_yaml_sections():
            lines.append(_SECTION_HEADERS[title])
            lines.append(body)
            lines.append("")
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    @staticmethod
    def parse_yaml(yaml_content: str) -> dict[str, Any]:
        """Parse YAML content into a dict of overrides.

        Uses ``yaml.safe_load``, so all YAML syntax is supported — including
        flow collections (``[a, b]``, ``{k: v}``) and block collections — for
        the list and map leaves (``labels``, ``annotations``,
        ``environment_variables``). Commented lines are ignored (template
        entries stay as comments until the user uncomments them).
        """
        import yaml

        data = yaml.safe_load(yaml_content)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"expected YAML mapping at top level, got {type(data).__name__}")
        return {k: (UNSET if v == _YAML_UNSET_TOKEN else v) for k, v in data.items()}

    @syncify
    @classmethod
    async def get_settings_for_edit(cls, project: str | None = None, domain: str | None = None) -> Settings:
        """Retrieve settings at the requested scope along with parent scopes.

        Returns a Settings object containing both the effective (resolved) settings
        with inheritance, and the local overrides at the requested scope.

        The scope is determined by ``domain`` and ``project``:

        - no args → ORG scope.
        - ``domain`` only → DOMAIN scope.
        - ``domain`` + ``project`` → PROJECT scope, inherits from DOMAIN.

        These are explicit parameters — they are **not** inferred from
        ``flyte.init()``.

        :param domain: Domain name.
        :param project: Project name. Requires ``domain`` to also be set.
        :returns: Settings object with effective_settings, local_settings, and version.

        Example::

            # Domain-level settings
            settings = Settings.get_settings_for_edit(domain="production")

            # Project-level — inherits from DOMAIN
            settings = Settings.get_settings_for_edit(domain="production", project="ml-pipeline")
        """
        ensure_client()
        cfg = get_init_config()
        client = get_client()

        key = settings_definition_pb2.SettingsKey(org=cfg.org or "", domain=domain or "", project=project or "")
        resp = await client.settings_service.get_settings_for_edit(
            settings_service_pb2.GetSettingsForEditRequest(key=key)
        )

        # resp.levels is ordered broadest → most specific.
        # Build effective settings by layering: broader values get overridden by narrower.
        # Also track parent_effective — the effective values from all levels except the local
        # scope — so each local override can show what it is replacing.
        want_domain = domain or ""
        want_project = project or ""

        effective: dict[str, EffectiveSetting] = {}
        parent_effective: dict[str, EffectiveSetting] = {}
        map_entry_origins: dict[str, dict[str, EffectiveSetting]] = {}
        list_item_origins: dict[str, list[EffectiveSetting]] = {}
        for record in resp.levels:
            is_local = (record.key.domain or "") == want_domain and (record.key.project or "") == want_project
            origin = _scope_origin_from_key(record.key)
            for dotkey, val in _proto_to_flat(record.settings):
                if not is_local:
                    parent_effective[dotkey] = EffectiveSetting(key=dotkey, value=val, origin=origin)
                    if _LEAF_TYPES.get(dotkey) == "stringmap":
                        if val is UNSET:
                            map_entry_origins.pop(dotkey, None)
                        elif isinstance(val, dict):
                            entry_map = map_entry_origins.setdefault(dotkey, {})
                            for entry_key, entry_val in val.items():
                                entry_map[entry_key] = EffectiveSetting(key=dotkey, value=entry_val, origin=origin)
                    elif _LEAF_TYPES.get(dotkey) == "stringlist":
                        if val is UNSET:
                            list_item_origins.pop(dotkey, None)
                        elif isinstance(val, list):
                            item_map = {es.value: es for es in list_item_origins.get(dotkey, [])}
                            for item in val:
                                item_map[item] = EffectiveSetting(key=dotkey, value=item, origin=origin)
                            list_item_origins[dotkey] = list(item_map.values())
                effective[dotkey] = EffectiveSetting(key=dotkey, value=val, origin=origin)

        # Local settings come from the level whose key equals the requested scope
        # exactly — that is the only level ``update_settings`` will ever write to.
        # We don't take ``levels[-1]`` because the server may omit a level when no
        # record exists at that scope yet, in which case the last returned level
        # is a parent scope and its values are inherited, not local.
        local_settings: list[LocalSetting] = []
        version = 0
        for record in resp.levels:
            if (record.key.domain or "") == want_domain and (record.key.project or "") == want_project:
                version = record.version
                for dotkey, val in _proto_to_flat(record.settings):
                    local_settings.append(LocalSetting(key=dotkey, value=val))
                break

        return cls(
            effective_settings=list(effective.values()),
            local_settings=local_settings,
            domain=domain,
            project=project,
            _version=version,
            _parent_effective=parent_effective,
            _map_entry_origins=map_entry_origins,
            _list_item_origins=list_item_origins,
        )

    @syncify
    async def update_settings(self, overrides: dict[str, Any]) -> None:
        """Replace the complete set of local overrides for this scope.

        Uses the scope (``domain`` / ``project``) this object was retrieved for.
        Settings not included in ``overrides`` will inherit from the parent scope.

        Uses optimistic locking via the version obtained from
        ``get_settings_for_edit``.

        :param overrides: Dict of flat dot-notation keys to values.
            Example: ``{"run.default_queue": "gpu", "security.service_account": "my-sa"}``

        Example::

            settings = Settings.get_settings_for_edit(domain="production")
            settings.update_settings({
                "run.default_queue": "gpu",
                "task_resource.min.cpu": "2",
            })
        """
        ensure_client()
        cfg = get_init_config()
        client = get_client()

        key = settings_definition_pb2.SettingsKey(
            org=cfg.org or "", domain=self.domain or "", project=self.project or ""
        )
        settings_proto = _flat_to_proto(overrides)

        if self._version == 0:
            req = settings_service_pb2.CreateSettingsRequest(key=key, settings=settings_proto)
            resp = await client.settings_service.create_settings(req)
        else:
            req = settings_service_pb2.UpdateSettingsRequest(
                key=key,
                settings=settings_proto,
                version=self._version,
            )
            resp = await client.settings_service.update_settings(req)

        # Refresh local state from the server response.
        self.local_settings = [LocalSetting(key=k, value=v) for k, v in overrides.items()]
        if resp.HasField("settingsRecord"):
            self._version = resp.settingsRecord.version
