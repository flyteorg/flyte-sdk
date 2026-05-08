"""Tests for Settings.get_settings_for_edit, Settings.update_settings, and proto conversion helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.settings import settings_definition_pb2, settings_service_pb2

from flyte.remote._settings import (
    _LEAF_DESCRIPTIONS,
    _LEAF_EXAMPLES,
    _LEAF_SCHEMA,
    UNSET,
    EffectiveSetting,
    LocalSetting,
    SettingOrigin,
    Settings,
    _build_leaf,
    _extract_leaf_value,
    _flat_to_proto,
    _proto_to_flat,
    _scope_origin_from_key,
    _walk_leaf,
)

# ---------------------------------------------------------------------------
# Proto conversion helpers
# ---------------------------------------------------------------------------


def _sv_string(s: str) -> settings_definition_pb2.StringSetting:
    return settings_definition_pb2.StringSetting(state=settings_definition_pb2.SETTING_STATE_VALUE, string_value=s)


def _sv_int(v: int) -> settings_definition_pb2.Int64Setting:
    return settings_definition_pb2.Int64Setting(state=settings_definition_pb2.SETTING_STATE_VALUE, int_value=v)


def _sv_bool(v: bool) -> settings_definition_pb2.BoolSetting:
    return settings_definition_pb2.BoolSetting(state=settings_definition_pb2.SETTING_STATE_VALUE, bool_value=v)


def _sv_quantity(v: str) -> settings_definition_pb2.QuantitySetting:
    return settings_definition_pb2.QuantitySetting(state=settings_definition_pb2.SETTING_STATE_VALUE, quantity_value=v)


def _sv_inherit_string() -> settings_definition_pb2.StringSetting:
    return settings_definition_pb2.StringSetting(state=settings_definition_pb2.SETTING_STATE_INHERIT)


class TestExtractLeafValue:
    def test_string(self):
        assert _extract_leaf_value(_sv_string("gpu"), "string") == "gpu"

    def test_int(self):
        assert _extract_leaf_value(_sv_int(42), "int") == 42

    def test_bool_true(self):
        assert _extract_leaf_value(_sv_bool(True), "bool") is True

    def test_bool_false(self):
        assert _extract_leaf_value(_sv_bool(False), "bool") is False

    def test_quantity(self):
        assert _extract_leaf_value(_sv_quantity("2"), "quantity") == "2"

    def test_stringlist(self):
        leaf = settings_definition_pb2.StringListSetting(
            state=settings_definition_pb2.SETTING_STATE_VALUE,
            list_value=settings_definition_pb2.StringValues(values=["a", "b"]),
        )
        assert _extract_leaf_value(leaf, "stringlist") == ["a", "b"]

    def test_stringmap(self):
        leaf = settings_definition_pb2.StringMapSetting(
            state=settings_definition_pb2.SETTING_STATE_VALUE,
            map_value=settings_definition_pb2.StringMap(entries={"k": "v"}),
        )
        assert _extract_leaf_value(leaf, "stringmap") == {"k": "v"}

    def test_inherit_returns_none(self):
        assert _extract_leaf_value(_sv_inherit_string(), "string") is None

    def test_default_state_returns_none(self):
        leaf = settings_definition_pb2.StringSetting()  # state defaults to INHERIT (0)
        assert _extract_leaf_value(leaf, "string") is None

    def test_unset_state_returns_unset_sentinel(self):
        leaf = settings_definition_pb2.StringSetting(state=settings_definition_pb2.SETTING_STATE_UNSET)
        assert _extract_leaf_value(leaf, "string") is UNSET

    def test_unset_sentinel_works_for_all_leaf_types(self):
        for cls, kind in [
            (settings_definition_pb2.Int64Setting, "int"),
            (settings_definition_pb2.BoolSetting, "bool"),
            (settings_definition_pb2.QuantitySetting, "quantity"),
            (settings_definition_pb2.StringListSetting, "stringlist"),
            (settings_definition_pb2.StringMapSetting, "stringmap"),
        ]:
            leaf = cls(state=settings_definition_pb2.SETTING_STATE_UNSET)
            assert _extract_leaf_value(leaf, kind) is UNSET, f"expected UNSET for {kind}"


class TestBuildLeaf:
    def test_string(self):
        leaf = _build_leaf("string", "gpu")
        assert isinstance(leaf, settings_definition_pb2.StringSetting)
        assert leaf.state == settings_definition_pb2.SETTING_STATE_VALUE
        assert leaf.string_value == "gpu"

    def test_int(self):
        leaf = _build_leaf("int", 42)
        assert isinstance(leaf, settings_definition_pb2.Int64Setting)
        assert leaf.int_value == 42

    def test_bool(self):
        leaf = _build_leaf("bool", True)
        assert isinstance(leaf, settings_definition_pb2.BoolSetting)
        assert leaf.bool_value is True

    def test_quantity_coerces_to_str(self):
        leaf = _build_leaf("quantity", 2)
        assert isinstance(leaf, settings_definition_pb2.QuantitySetting)
        assert leaf.quantity_value == "2"

    def test_stringlist(self):
        leaf = _build_leaf("stringlist", ["a", "b"])
        assert isinstance(leaf, settings_definition_pb2.StringListSetting)
        assert list(leaf.list_value.values) == ["a", "b"]

    def test_stringmap(self):
        leaf = _build_leaf("stringmap", {"k": "v"})
        assert isinstance(leaf, settings_definition_pb2.StringMapSetting)
        assert dict(leaf.map_value.entries) == {"k": "v"}

    def test_stringlist_requires_sequence(self):
        with pytest.raises(TypeError):
            _build_leaf("stringlist", "not-a-list")

    def test_stringmap_requires_dict(self):
        with pytest.raises(TypeError):
            _build_leaf("stringmap", ["not", "a", "dict"])

    def test_build_unset_string(self):
        leaf = _build_leaf("string", UNSET)
        assert isinstance(leaf, settings_definition_pb2.StringSetting)
        assert leaf.state == settings_definition_pb2.SETTING_STATE_UNSET

    def test_build_unset_int(self):
        leaf = _build_leaf("int", UNSET)
        assert isinstance(leaf, settings_definition_pb2.Int64Setting)
        assert leaf.state == settings_definition_pb2.SETTING_STATE_UNSET

    def test_build_unset_bool(self):
        leaf = _build_leaf("bool", UNSET)
        assert isinstance(leaf, settings_definition_pb2.BoolSetting)
        assert leaf.state == settings_definition_pb2.SETTING_STATE_UNSET

    def test_build_unset_quantity(self):
        leaf = _build_leaf("quantity", UNSET)
        assert isinstance(leaf, settings_definition_pb2.QuantitySetting)
        assert leaf.state == settings_definition_pb2.SETTING_STATE_UNSET

    def test_build_unset_stringlist(self):
        leaf = _build_leaf("stringlist", UNSET)
        assert isinstance(leaf, settings_definition_pb2.StringListSetting)
        assert leaf.state == settings_definition_pb2.SETTING_STATE_UNSET

    def test_build_unset_stringmap(self):
        leaf = _build_leaf("stringmap", UNSET)
        assert isinstance(leaf, settings_definition_pb2.StringMapSetting)
        assert leaf.state == settings_definition_pb2.SETTING_STATE_UNSET


class TestProtoFlatRoundtrip:
    def test_all_leaf_types_roundtrip(self):
        overrides = {
            "run.default_queue": "gpu",
            "security.service_account": "ml-sa",
            "storage.raw_data_path": "s3://bucket/data",
            "task_resource.min.cpu": "2",
            "task_resource.max.memory": "8Gi",
            "task_resource.mirror_limits_request": True,
            "labels": {"env": "prod", "team": "ml"},
            "annotations": {"oncall": "ml-team"},
            "environment_variables": {"DEBUG": "0"},
        }
        proto = _flat_to_proto(overrides)
        wire = proto.SerializeToString()
        restored = settings_definition_pb2.Settings.FromString(wire)

        flat = dict(_proto_to_flat(restored))
        assert flat == overrides

    def test_flat_to_proto_sets_nested_fields(self):
        proto = _flat_to_proto({"task_resource.min.cpu": "2"})
        assert proto.HasField("task_resource")
        assert proto.task_resource.HasField("min")
        assert proto.task_resource.min.HasField("cpu")
        assert proto.task_resource.min.cpu.quantity_value == "2"
        assert proto.task_resource.min.cpu.state == settings_definition_pb2.SETTING_STATE_VALUE

    def test_flat_to_proto_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown settings key"):
            _flat_to_proto({"not.a.real.key": "x"})

    def test_proto_to_flat_skips_inherited(self):
        proto = settings_definition_pb2.Settings(
            run=settings_definition_pb2.RunSettings(
                default_queue=_sv_string("gpu"),
            ),
            security=settings_definition_pb2.SecuritySettings(
                service_account=settings_definition_pb2.StringSetting(
                    state=settings_definition_pb2.SETTING_STATE_INHERIT,
                ),
            ),
        )
        flat = dict(_proto_to_flat(proto))
        assert flat == {"run.default_queue": "gpu"}

    def test_proto_to_flat_includes_unset(self):
        proto = settings_definition_pb2.Settings(
            run=settings_definition_pb2.RunSettings(
                default_queue=settings_definition_pb2.StringSetting(state=settings_definition_pb2.SETTING_STATE_UNSET),
            ),
            security=settings_definition_pb2.SecuritySettings(
                service_account=_sv_string("ml-sa"),
            ),
        )
        flat = dict(_proto_to_flat(proto))
        assert flat["run.default_queue"] is UNSET
        assert flat["security.service_account"] == "ml-sa"

    def test_flat_to_proto_with_unset_value(self):
        proto = _flat_to_proto({"run.default_queue": UNSET})
        assert proto.run.default_queue.state == settings_definition_pb2.SETTING_STATE_UNSET

    def test_unset_roundtrip(self):
        overrides = {"run.default_queue": UNSET, "security.service_account": "ml-sa"}
        proto = _flat_to_proto(overrides)
        wire = proto.SerializeToString()
        restored = settings_definition_pb2.Settings.FromString(wire)
        flat = dict(_proto_to_flat(restored))
        assert flat["run.default_queue"] is UNSET
        assert flat["security.service_account"] == "ml-sa"


class TestWalkLeaf:
    def test_returns_none_when_parent_unset(self):
        proto = settings_definition_pb2.Settings()
        assert _walk_leaf(proto, "run.default_queue") is None

    def test_navigates_nested_path(self):
        proto = _flat_to_proto({"task_resource.min.cpu": "4"})
        leaf = _walk_leaf(proto, "task_resource.min.cpu")
        assert leaf is not None
        assert leaf.quantity_value == "4"


class TestScopeOriginFromKey:
    def test_org_scope(self):
        key = settings_definition_pb2.SettingsKey(org="myorg")
        origin = _scope_origin_from_key(key)
        assert origin.scope_type == "ORG"

    def test_domain_scope(self):
        key = settings_definition_pb2.SettingsKey(org="myorg", domain="production")
        origin = _scope_origin_from_key(key)
        assert origin.scope_type == "DOMAIN"
        assert origin.domain == "production"

    def test_project_scope(self):
        key = settings_definition_pb2.SettingsKey(org="myorg", domain="production", project="ml")
        origin = _scope_origin_from_key(key)
        assert origin.scope_type == "PROJECT"
        assert origin.domain == "production"
        assert origin.project == "ml"


# ---------------------------------------------------------------------------
# Settings YAML
# ---------------------------------------------------------------------------


class TestToYaml:
    def test_local_inherited_and_template(self):
        settings = Settings(
            effective_settings=[
                EffectiveSetting(key="run.default_queue", value="gpu", origin=SettingOrigin("PROJECT", "prod", "ml")),
                EffectiveSetting(
                    key="storage.raw_data_path", value="s3://bucket/data", origin=SettingOrigin("DOMAIN", "prod")
                ),
                EffectiveSetting(key="security.service_account", value="sa", origin=SettingOrigin("ORG")),
            ],
            local_settings=[
                LocalSetting(key="run.default_queue", value="gpu"),
            ],
            domain="prod",
            project="ml",
        )
        yaml = settings.to_yaml()
        # Three prefix tiers: ### = headers, ## = docs/meta, # = commented setting.
        assert "### Settings for scope: PROJECT(prod/ml)" in yaml
        assert "### Local overrides" in yaml
        assert "### Inherited settings" in yaml
        assert "### Available settings" in yaml
        assert "run.default_queue: gpu" in yaml
        assert '# storage.raw_data_path: "s3://bucket/data"' in yaml
        assert "# security.service_account: sa" in yaml
        assert "## inherited from DOMAIN(prod)" in yaml
        assert "## inherited from ORG" in yaml
        for key in Settings.available_keys():
            if key in {"run.default_queue", "storage.raw_data_path", "security.service_account"}:
                continue
            assert f"# {key}: {_LEAF_EXAMPLES[key]}" in yaml

    def test_no_local_settings(self):
        settings = Settings(
            effective_settings=[
                EffectiveSetting(key="run.default_queue", value="cpu", origin=SettingOrigin("ORG")),
            ],
            local_settings=[],
            domain="prod",
        )
        yaml = settings.to_yaml()
        assert "### Local overrides" not in yaml
        assert "# run.default_queue: cpu" in yaml
        assert "### Available settings" in yaml

    def test_empty_shows_full_template(self):
        """Even when nothing is set, every available key is listed as a commented example."""
        settings = Settings(effective_settings=[], local_settings=[])
        yaml = settings.to_yaml()
        assert "### Local overrides" not in yaml
        assert "### Inherited settings" not in yaml
        assert "### Available settings" in yaml
        for key in Settings.available_keys():
            assert f"# {key}: {_LEAF_EXAMPLES[key]}" in yaml

    def test_yaml_template_round_trips_through_parser(self):
        """An unedited template must parse back to an empty override set."""
        settings = Settings(effective_settings=[], local_settings=[])
        yaml = settings.to_yaml()
        assert Settings.parse_yaml(yaml) == {}

    def test_each_field_description_appears_above_its_key(self):
        """Descriptions pulled from the proto ``desc`` extension appear as
        ``##``-prefixed comment lines immediately above the corresponding key."""
        settings = Settings(
            effective_settings=[
                EffectiveSetting(key="run.default_queue", value="gpu", origin=SettingOrigin("DOMAIN", "prod")),
            ],
            local_settings=[LocalSetting(key="security.service_account", value="ml-sa")],
            domain="prod",
        )
        yaml = settings.to_yaml()
        for dotkey, description in _LEAF_DESCRIPTIONS.items():
            if not description:
                continue
            lines = yaml.split("\n")
            key_indices = [i for i, ln in enumerate(lines) if ln.lstrip("# ").startswith(f"{dotkey}:")]
            assert key_indices, f"key {dotkey!r} not rendered"
            for idx in key_indices:
                assert lines[idx - 1] == f"## {description}", (
                    f"expected description above {dotkey!r}; got {lines[idx - 1]!r}"
                )

    def test_inherited_map_renders_one_line_per_entry(self):
        settings = Settings(
            effective_settings=[
                EffectiveSetting(
                    key="annotations",
                    value={"team": "ml", "env": "prod"},
                    origin=SettingOrigin("DOMAIN", "prod"),
                ),
            ],
            local_settings=[],
            domain="prod",
            project="ml",
        )
        yaml = settings.to_yaml()
        assert "#   team: ml" in yaml
        assert "#   env: prod" in yaml
        assert "## inherited from DOMAIN(prod)" in yaml
        assert "{team:" not in yaml and "team: ml," not in yaml

    def test_map_with_only_inherited_entries_appears_in_inherited_section(self):
        """A local map whose value is empty (all entries from parent) belongs in Inherited, not Local overrides."""
        settings = Settings(
            effective_settings=[
                EffectiveSetting(key="annotations", value={"team": "ml"}, origin=SettingOrigin("DOMAIN", "prod")),
            ],
            local_settings=[LocalSetting(key="annotations", value={})],
            domain="prod",
            project="ml",
            _map_entry_origins={
                "annotations": {
                    "team": EffectiveSetting(key="annotations", value="ml", origin=SettingOrigin("DOMAIN", "prod")),
                }
            },
        )
        yaml = settings.to_yaml()
        lines = yaml.split("\n")
        inherited_idx = next(i for i, line in enumerate(lines) if "### Inherited settings" in line)
        available_idx = next(i for i, line in enumerate(lines) if "### Available settings" in line)
        annotations_idx = next(i for i, line in enumerate(lines) if "annotations:" in line)
        assert inherited_idx < annotations_idx < available_idx, (
            "annotations should appear in Inherited settings, not Local overrides"
        )
        assert "### Local overrides" not in yaml

    def test_map_with_only_inherited_entries_comments_out_key_line(self):
        """If a local map has no local entries (only parent entries), the key line itself must be
        commented out so that saving the file unchanged does not replace the map with null."""
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="annotations", value={})],
            domain="prod",
            project="ml",
            _map_entry_origins={
                "annotations": {
                    "team": EffectiveSetting(key="annotations", value="ml", origin=SettingOrigin("DOMAIN", "prod")),
                }
            },
        )
        yaml = settings.to_yaml()
        lines = yaml.split("\n")
        assert not any(line.startswith("annotations:") for line in lines), "uncommented key line must not appear"
        assert any(line.startswith("# annotations:") for line in lines)
        assert Settings.parse_yaml(yaml).get("annotations") is None

    def test_local_map_renders_as_block_yaml_not_flow(self):
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="annotations", value={"oncall": "ml-team"})],
            domain="prod",
        )
        yaml = settings.to_yaml()
        assert "annotations:" in yaml
        assert "  oncall: ml-team" in yaml
        assert "annotations: {" not in yaml

    def test_local_map_parent_entries_commented_with_origin(self):
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="annotations", value={"oncall": "ml-team"})],
            domain="prod",
            project="ml",
            _map_entry_origins={
                "annotations": {
                    "team": EffectiveSetting(key="annotations", value="ml", origin=SettingOrigin("DOMAIN", "prod")),
                    "env": EffectiveSetting(key="annotations", value="prod", origin=SettingOrigin("ORG")),
                }
            },
        )
        yaml = settings.to_yaml()
        assert "  # team: ml  ## defined at DOMAIN(prod)" in yaml
        assert "  # env: prod  ## defined at ORG" in yaml

    def test_local_map_local_key_not_duplicated_as_comment(self):
        """A map entry key set locally must not also appear as a commented parent line."""
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="annotations", value={"team": "engineering"})],
            domain="prod",
            project="ml",
            _map_entry_origins={
                "annotations": {
                    "team": EffectiveSetting(key="annotations", value="ml", origin=SettingOrigin("ORG")),
                }
            },
        )
        yaml = settings.to_yaml()
        assert "  team: engineering" in yaml
        assert "  # team:" not in yaml

    def test_local_map_how_maps_work_comment_present(self):
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="annotations", value={"oncall": "ml-team"})],
            domain="prod",
        )
        yaml = settings.to_yaml()
        assert "Map entries merge across scopes" in yaml

    def test_local_map_no_parent_entries_still_block_format(self):
        """Block format is used even when there are no parent entries to show."""
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="annotations", value={"oncall": "ml-team"})],
            domain="prod",
        )
        yaml = settings.to_yaml()
        assert "annotations:" in yaml
        assert "  oncall: ml-team" in yaml

    def test_local_override_shows_overridden_value_comment(self):
        settings = Settings(
            effective_settings=[
                EffectiveSetting(key="run.default_queue", value="gpu", origin=SettingOrigin("PROJECT", "prod", "ml")),
            ],
            local_settings=[LocalSetting(key="run.default_queue", value="gpu")],
            domain="prod",
            project="ml",
            _parent_effective={
                "run.default_queue": EffectiveSetting(
                    key="run.default_queue", value="cpu", origin=SettingOrigin("DOMAIN", "prod")
                )
            },
        )
        yaml = settings.to_yaml()
        lines = yaml.split("\n")
        local_line_idx = next(i for i, line in enumerate(lines) if line == "run.default_queue: gpu")
        assert lines[local_line_idx + 1] == "# overridden value: cpu  ## defined at DOMAIN(prod)"

    def test_local_override_no_parent_shows_no_overridden_comment(self):
        settings = Settings(
            effective_settings=[
                EffectiveSetting(key="run.default_queue", value="gpu", origin=SettingOrigin("PROJECT", "prod", "ml")),
            ],
            local_settings=[LocalSetting(key="run.default_queue", value="gpu")],
            domain="prod",
            project="ml",
        )
        yaml = settings.to_yaml()
        assert "# overridden value:" not in yaml

    def test_local_unset_renders_as_tilde_unset(self):
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="run.default_queue", value=UNSET)],
            domain="prod",
        )
        yaml = settings.to_yaml()
        assert "run.default_queue: ~unset" in yaml

    def test_unset_local_setting_appears_in_local_section_not_available(self):
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="run.default_queue", value=UNSET)],
            domain="prod",
        )
        yaml = settings.to_yaml()
        assert "### Local overrides" in yaml
        lines = yaml.split("\n")
        local_idx = next(i for i, line in enumerate(lines) if "### Local overrides" in line)
        available_idx = next(i for i, line in enumerate(lines) if "### Available settings" in line)
        queue_idx = next(i for i, line in enumerate(lines) if "run.default_queue: ~unset" in line)
        assert local_idx < queue_idx < available_idx

    def test_bulk_uncomment_activates_only_valid_settings(self):
        """Stripping a single leading ``# `` from every line should activate every
        commented setting without producing garbage from documentation lines."""
        settings = Settings(effective_settings=[], local_settings=[], domain="prod")
        yaml = settings.to_yaml()
        # Remove one leading '# ' from every line (the common bulk-uncomment gesture).
        uncommented = "\n".join((ln.removeprefix("# ")) for ln in yaml.split("\n"))
        overrides = Settings.parse_yaml(uncommented)
        # Every available key should now be a live override; descriptions stay
        # as '#' comments (one hash survived from the original '##').
        assert set(overrides.keys()) == set(Settings.available_keys())
        # And the result is valid input for the proto builder.
        _flat_to_proto(overrides)


class TestParseYaml:
    def test_parse_overrides(self):
        yaml_content = """# Local overrides
run.default_queue: gpu
task_resource.min.cpu: '2'
security.service_account: my-sa

# Inherited settings (uncomment to override)
# storage.raw_data_path: s3://bucket  # inherited from ORG
"""
        overrides = Settings.parse_yaml(yaml_content)
        assert overrides == {
            "run.default_queue": "gpu",
            "task_resource.min.cpu": "2",
            "security.service_account": "my-sa",
        }

    def test_parse_bool_values(self):
        overrides = Settings.parse_yaml("task_resource.mirror_limits_request: true\n")
        assert overrides["task_resource.mirror_limits_request"] is True

    def test_parse_empty(self):
        assert Settings.parse_yaml("") == {}
        assert Settings.parse_yaml("# only comments\n# here") == {}

    def test_parse_strips_inline_comments(self):
        overrides = Settings.parse_yaml("run.default_queue: gpu  # my queue\n")
        assert overrides["run.default_queue"] == "gpu"

    def test_parse_quoted_string(self):
        overrides = Settings.parse_yaml('run.default_queue: "123"\n')
        assert overrides["run.default_queue"] == "123"

    def test_parse_flow_list(self):
        overrides = Settings.parse_yaml("labels: ['env:prod', 'team:ml']\n")
        assert overrides["labels"] == ["env:prod", "team:ml"]

    def test_parse_flow_map(self):
        overrides = Settings.parse_yaml("annotations: {oncall: ml-team}\n")
        assert overrides["annotations"] == {"oncall": "ml-team"}

    def test_parse_template_uncomment(self):
        """Generic placeholders, once uncommented, feed straight through _flat_to_proto."""
        uncommented = "".join(f"{k}: {_LEAF_EXAMPLES[k]}\n" for k in Settings.available_keys())
        overrides = Settings.parse_yaml(uncommented)
        # Every leaf placeholder must be acceptable input to the proto builder.
        _flat_to_proto(overrides)

    def test_parse_non_mapping_raises(self):
        with pytest.raises(ValueError):
            Settings.parse_yaml("- just\n- a\n- list\n")

    def test_parse_tilde_unset_returns_unset_sentinel(self):
        overrides = Settings.parse_yaml("run.default_queue: ~unset\n")
        assert overrides["run.default_queue"] is UNSET

    def test_parse_tilde_unset_roundtrips_to_proto(self):
        overrides = Settings.parse_yaml("run.default_queue: ~unset\nsecurity.service_account: ml-sa\n")
        assert overrides["run.default_queue"] is UNSET
        proto = _flat_to_proto(overrides)
        assert proto.run.default_queue.state == settings_definition_pb2.SETTING_STATE_UNSET
        assert proto.security.service_account.string_value == "ml-sa"

    def test_parse_tilde_unset_only_in_values_not_keys(self):
        overrides = Settings.parse_yaml("security.service_account: ml-sa\n")
        assert "~unset" not in overrides


# ---------------------------------------------------------------------------
# Settings.get_settings_for_edit and Settings.update_settings (RPC integration)
# ---------------------------------------------------------------------------


def _make_record(
    org: str = "",
    domain: str = "",
    project: str = "",
    version: int = 1,
    **setting_values,
) -> settings_service_pb2.SettingsRecord:
    """Build a SettingsRecord with the given flat dot-notation overrides."""
    key = settings_definition_pb2.SettingsKey(org=org, domain=domain, project=project)
    return settings_service_pb2.SettingsRecord(
        key=key,
        settings=_flat_to_proto(dict(setting_values)),
        version=version,
    )


@pytest.fixture
def mock_settings_service():
    svc = MagicMock()
    svc.get_settings_for_edit = AsyncMock()
    svc.update_settings = AsyncMock(return_value=settings_service_pb2.UpdateSettingsResponse())
    svc.create_settings = AsyncMock(return_value=settings_service_pb2.CreateSettingsResponse())
    return svc


@pytest.fixture
def mock_client(mock_settings_service):
    client = MagicMock()
    client.settings_service = mock_settings_service
    return client


@pytest.fixture
def mock_init_config():
    cfg = MagicMock()
    cfg.org = "myorg"
    cfg.project = "default-proj"
    cfg.domain = "default-domain"
    return cfg


_PATCH_ENSURE = "flyte.remote._settings.ensure_client"
_PATCH_CLIENT = "flyte.remote._settings.get_client"
_PATCH_CONFIG = "flyte.remote._settings.get_init_config"


class TestSettingsGet:
    def test_get_org_scope(self, mock_client, mock_settings_service, mock_init_config):
        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg"),
            levels=[
                _make_record(
                    org="myorg",
                    version=5,
                    **{"run.default_queue": "cpu", "security.service_account": "default-sa"},
                ),
            ],
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get_settings_for_edit()

        assert len(settings.effective_settings) == 2
        assert len(settings.local_settings) == 2
        assert settings._version == 5
        assert settings.domain is None
        assert settings.project is None

        effective_keys = {s.key for s in settings.effective_settings}
        assert "run.default_queue" in effective_keys
        assert "security.service_account" in effective_keys

    def test_get_project_scope_with_inheritance(self, mock_client, mock_settings_service, mock_init_config):
        org_record = _make_record(
            org="myorg",
            version=1,
            **{"run.default_queue": "cpu", "security.service_account": "org-sa"},
        )
        domain_record = _make_record(
            org="myorg",
            domain="prod",
            version=3,
            **{"run.default_queue": "gpu"},
        )
        project_record = _make_record(
            org="myorg",
            domain="prod",
            project="ml",
            version=7,
            **{"security.service_account": "ml-sa"},
        )

        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg", domain="prod", project="ml"),
            levels=[org_record, domain_record, project_record],
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get_settings_for_edit(domain="prod", project="ml")

        assert settings._version == 7
        assert settings.domain == "prod"
        assert settings.project == "ml"

        effective = {s.key: s for s in settings.effective_settings}
        assert effective["run.default_queue"].value == "gpu"
        assert effective["run.default_queue"].origin.scope_type == "DOMAIN"
        assert effective["security.service_account"].value == "ml-sa"
        assert effective["security.service_account"].origin.scope_type == "PROJECT"

        local_keys = {s.key for s in settings.local_settings}
        assert local_keys == {"security.service_account"}

    def test_get_sends_correct_key(self, mock_client, mock_settings_service, mock_init_config):
        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            levels=[]
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            Settings.get_settings_for_edit(domain="staging", project="web")

        req = mock_settings_service.get_settings_for_edit.call_args[0][0]
        assert req.key.org == "myorg"
        assert req.key.domain == "staging"
        assert req.key.project == "web"

    def test_get_local_matches_requested_scope_not_last_level(
        self, mock_client, mock_settings_service, mock_init_config
    ):
        """When the server returns no record at the requested scope, nothing is local."""
        # User asks for PROJECT, but server returns only ORG and DOMAIN — no project record exists yet.
        org_record = _make_record(org="myorg", version=1, **{"run.default_queue": "cpu"})
        domain_record = _make_record(org="myorg", domain="prod", version=3, **{"run.default_queue": "gpu"})

        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg", domain="prod", project="ml"),
            levels=[org_record, domain_record],
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get_settings_for_edit(domain="prod", project="ml")

        # DOMAIN values are inherited, not local — the requested PROJECT scope has no record.
        assert settings.local_settings == []
        assert settings._version == 0
        # Effective still surfaces the DOMAIN-level value.
        assert any(s.key == "run.default_queue" and s.value == "gpu" for s in settings.effective_settings)

    def test_get_populates_map_entry_origins_from_parent_levels(
        self, mock_client, mock_settings_service, mock_init_config
    ):
        """Each parent-scope map entry is recorded with the scope it came from."""
        org_record = _make_record(org="myorg", version=1, annotations={"team": "ml", "env": "prod"})
        project_record = _make_record(
            org="myorg", domain="prod", project="ml", version=2, annotations={"oncall": "ml-team"}
        )
        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg", domain="prod", project="ml"),
            levels=[org_record, project_record],
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get_settings_for_edit(domain="prod", project="ml")

        origins = settings._map_entry_origins["annotations"]
        assert origins["team"].value == "ml"
        assert origins["team"].origin.scope_type == "ORG"
        assert origins["env"].value == "prod"
        assert origins["env"].origin.scope_type == "ORG"

    def test_get_map_entry_origins_excludes_local_scope_entries(
        self, mock_client, mock_settings_service, mock_init_config
    ):
        """Entries that exist only at the local scope do not appear in map_entry_origins."""
        project_record = _make_record(
            org="myorg", domain="prod", project="ml", version=1, annotations={"oncall": "ml-team"}
        )
        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg", domain="prod", project="ml"),
            levels=[project_record],
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get_settings_for_edit(domain="prod", project="ml")

        assert "annotations" not in settings._map_entry_origins

    def test_get_map_entry_origins_narrower_parent_wins_on_shared_entry_key(
        self, mock_client, mock_settings_service, mock_init_config
    ):
        """If ORG and DOMAIN both set the same map entry key, the DOMAIN value appears in origins."""
        org_record = _make_record(org="myorg", version=1, annotations={"team": "ml"})
        domain_record = _make_record(org="myorg", domain="prod", version=2, annotations={"team": "engineering"})
        project_record = _make_record(
            org="myorg", domain="prod", project="ml", version=3, annotations={"oncall": "ml-team"}
        )
        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg", domain="prod", project="ml"),
            levels=[org_record, domain_record, project_record],
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get_settings_for_edit(domain="prod", project="ml")

        origins = settings._map_entry_origins["annotations"]
        assert origins["team"].value == "engineering"
        assert origins["team"].origin.scope_type == "DOMAIN"

    def test_get_populates_parent_effective_for_overridden_keys(
        self, mock_client, mock_settings_service, mock_init_config
    ):
        """parent_effective holds the value from the parent scope for keys overridden locally."""
        domain_record = _make_record(org="myorg", domain="prod", version=1, **{"run.default_queue": "cpu"})
        project_record = _make_record(
            org="myorg", domain="prod", project="ml", version=2, **{"run.default_queue": "gpu"}
        )
        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg", domain="prod", project="ml"),
            levels=[domain_record, project_record],
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get_settings_for_edit(domain="prod", project="ml")

        assert "run.default_queue" in settings._parent_effective
        parent = settings._parent_effective["run.default_queue"]
        assert parent.value == "cpu"
        assert parent.origin.scope_type == "DOMAIN"

    def test_get_parent_effective_empty_when_no_parent_sets_key(
        self, mock_client, mock_settings_service, mock_init_config
    ):
        """Keys set only at the local scope have no entry in parent_effective."""
        project_record = _make_record(
            org="myorg", domain="prod", project="ml", version=1, **{"run.default_queue": "gpu"}
        )
        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg", domain="prod", project="ml"),
            levels=[project_record],
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get_settings_for_edit(domain="prod", project="ml")

        assert "run.default_queue" not in settings._parent_effective

    def test_unset_at_parent_scope_clears_ancestor_map_entries(
        self, mock_client, mock_settings_service, mock_init_config
    ):
        """When a parent scope sets a map to UNSET, ancestor map entries must not appear in _map_entry_origins."""
        org_record = _make_record(org="myorg", version=1, annotations={"team": "ml", "env": "prod"})
        domain_record = _make_record(org="myorg", domain="prod", version=2, annotations=UNSET)
        project_record = _make_record(
            org="myorg", domain="prod", project="ml", version=3, annotations={"oncall": "ml-team"}
        )
        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg", domain="prod", project="ml"),
            levels=[org_record, domain_record, project_record],
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get_settings_for_edit(domain="prod", project="ml")

        origins = settings._map_entry_origins.get("annotations", {})
        assert "team" not in origins, "ORG entry 'team' should be blocked by DOMAIN's UNSET"
        assert "env" not in origins, "ORG entry 'env' should be blocked by DOMAIN's UNSET"

    def test_get_surfaces_unset_from_local_scope(self, mock_client, mock_settings_service, mock_init_config):
        """A SETTING_STATE_UNSET leaf in the server response appears as UNSET in local_settings."""
        unset_proto = settings_definition_pb2.Settings(
            run=settings_definition_pb2.RunSettings(
                default_queue=settings_definition_pb2.StringSetting(state=settings_definition_pb2.SETTING_STATE_UNSET)
            )
        )
        project_record = settings_service_pb2.SettingsRecord(
            key=settings_definition_pb2.SettingsKey(org="myorg", domain="prod", project="ml"),
            settings=unset_proto,
            version=2,
        )
        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=project_record.key,
            levels=[project_record],
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get_settings_for_edit(domain="prod", project="ml")

        local = {s.key: s.value for s in settings.local_settings}
        assert local["run.default_queue"] is UNSET

    def test_get_empty_levels(self, mock_client, mock_settings_service, mock_init_config):
        mock_settings_service.get_settings_for_edit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            levels=[]
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get_settings_for_edit()

        assert settings.effective_settings == []
        assert settings.local_settings == []
        assert settings._version == 0


class TestSettingsUpdate:
    def test_update_sends_correct_request(self, mock_client, mock_settings_service, mock_init_config):
        settings = Settings(
            effective_settings=[],
            local_settings=[],
            domain="prod",
            project="ml",
            _version=7,
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings.update_settings({"run.default_queue": "gpu", "security.service_account": "ml-sa"})

        mock_settings_service.update_settings.assert_called_once()
        req = mock_settings_service.update_settings.call_args[0][0]
        assert req.key.org == "myorg"
        assert req.key.domain == "prod"
        assert req.key.project == "ml"
        assert req.version == 7
        assert req.settings.run.default_queue.string_value == "gpu"
        assert req.settings.run.default_queue.state == settings_definition_pb2.SETTING_STATE_VALUE
        assert req.settings.security.service_account.string_value == "ml-sa"

    def test_update_refreshes_local_state(self, mock_client, mock_settings_service, mock_init_config):
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="old.key", value="old")],
            domain="prod",
            _version=3,
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings.update_settings({"run.default_queue": "gpu"})

        assert len(settings.local_settings) == 1
        assert settings.local_settings[0].key == "run.default_queue"
        assert settings.local_settings[0].value == "gpu"

    def test_update_empty_overrides_clears_scope(self, mock_client, mock_settings_service, mock_init_config):
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="run.default_queue", value="gpu")],
            _version=1,
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings.update_settings({})

        req = mock_settings_service.update_settings.call_args[0][0]
        # Empty overrides → empty Settings proto → no fields set
        assert req.settings.ByteSize() == 0
        assert settings.local_settings == []

    def test_update_with_unset_sends_unset_state_in_proto(self, mock_client, mock_settings_service, mock_init_config):
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="run.default_queue", value="gpu")],
            domain="prod",
            _version=4,
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings.update_settings({"run.default_queue": UNSET})

        req = mock_settings_service.update_settings.call_args[0][0]
        assert req.settings.run.default_queue.state == settings_definition_pb2.SETTING_STATE_UNSET

    def test_update_refreshes_local_state_with_unset(self, mock_client, mock_settings_service, mock_init_config):
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="run.default_queue", value="gpu")],
            domain="prod",
            _version=4,
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings.update_settings({"run.default_queue": UNSET})

        assert settings.local_settings[0].key == "run.default_queue"
        assert settings.local_settings[0].value is UNSET

    def test_update_picks_up_version_from_response(self, mock_client, mock_settings_service, mock_init_config):
        mock_settings_service.update_settings.return_value = settings_service_pb2.UpdateSettingsResponse(
            settingsRecord=settings_service_pb2.SettingsRecord(
                key=settings_definition_pb2.SettingsKey(org="myorg", domain="prod"),
                settings=_flat_to_proto({"run.default_queue": "gpu"}),
                version=99,
            )
        )
        settings = Settings(effective_settings=[], local_settings=[], domain="prod", _version=3)

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings.update_settings({"run.default_queue": "gpu"})

        assert settings._version == 99


# ---------------------------------------------------------------------------
# SettingOrigin display
# ---------------------------------------------------------------------------


class TestSettingOriginStr:
    def test_org(self):
        assert str(SettingOrigin("ORG")) == "ORG"

    def test_domain(self):
        assert str(SettingOrigin("DOMAIN", domain="prod")) == "DOMAIN(prod)"

    def test_project(self):
        assert str(SettingOrigin("PROJECT", domain="prod", project="ml")) == "PROJECT(prod/ml)"


# ---------------------------------------------------------------------------
# Programmatic helpers
# ---------------------------------------------------------------------------


class TestProgrammaticAccess:
    def test_available_keys_contains_known_leaves(self):
        keys = Settings.available_keys()
        assert "run.default_queue" in keys
        assert "task_resource.min.cpu" in keys
        assert "environment_variables" in keys

    def test_schema_is_derived_from_proto_descriptor(self):
        """The schema must come from walking Settings.DESCRIPTOR, not a hand-written list."""
        assert len(_LEAF_SCHEMA) > 0
        for dotkey, kind, field_descriptor in _LEAF_SCHEMA:
            assert kind in {"string", "int", "bool", "quantity", "stringlist", "stringmap"}
            # The field descriptor points at a *Setting message in the proto schema.
            assert field_descriptor.message_type is not None
            assert field_descriptor.message_type.name.endswith("Setting")
            # Last segment of the dotkey equals the proto field name.
            assert dotkey.rsplit(".", 1)[-1] == field_descriptor.name

    def test_every_leaf_has_a_description(self):
        """Every leaf in the proto schema carries a `desc` annotation."""
        missing = [k for k, d in _LEAF_DESCRIPTIONS.items() if not d]
        assert missing == [], f"leaves missing proto desc extension: {missing}"

    def test_local_overrides_and_effective_values(self):
        settings = Settings(
            effective_settings=[
                EffectiveSetting(key="run.default_queue", value="gpu", origin=SettingOrigin("PROJECT", "prod", "ml")),
                EffectiveSetting(
                    key="security.service_account", value="ml-sa", origin=SettingOrigin("DOMAIN", "prod")
                ),
            ],
            local_settings=[LocalSetting(key="run.default_queue", value="gpu")],
        )
        assert settings.local_overrides() == {"run.default_queue": "gpu"}
        assert settings.effective_values() == {"run.default_queue": "gpu", "security.service_account": "ml-sa"}
