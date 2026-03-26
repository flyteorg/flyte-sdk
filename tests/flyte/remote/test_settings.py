"""Tests for Settings.get, Settings.update, and proto conversion helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.org import settings_definition_pb2, settings_service_pb2

from flyte.remote._settings import (
    EffectiveSetting,
    LocalSetting,
    SettingOrigin,
    Settings,
    _extract_setting_value,
    _make_setting_value,
    _raw_data_to_flat,
    _scope_origin_from_key,
)

# ---------------------------------------------------------------------------
# Proto conversion helpers
# ---------------------------------------------------------------------------


class TestExtractSettingValue:
    def test_string_value(self):
        sv = settings_definition_pb2.SettingValue(string_value="gpu")
        assert _extract_setting_value(sv) == "gpu"

    def test_int_value(self):
        sv = settings_definition_pb2.SettingValue(int_value=42)
        assert _extract_setting_value(sv) == 42

    def test_bool_value(self):
        sv = settings_definition_pb2.SettingValue(bool_value=True)
        assert _extract_setting_value(sv) is True

    def test_list_value(self):
        sv = settings_definition_pb2.SettingValue(list_value=settings_definition_pb2.StringValues(values=["a", "b"]))
        assert _extract_setting_value(sv) == ["a", "b"]

    def test_map_value(self):
        sv = settings_definition_pb2.SettingValue(map_value=settings_definition_pb2.StringMap(entries={"k": "v"}))
        assert _extract_setting_value(sv) == {"k": "v"}

    def test_inherit_returns_none(self):
        sv = settings_definition_pb2.SettingValue(state=settings_definition_pb2.SETTING_STATE_INHERIT)
        assert _extract_setting_value(sv) is None

    def test_unset_returns_none(self):
        sv = settings_definition_pb2.SettingValue()
        assert _extract_setting_value(sv) is None


class TestSettingValueExclusivity:
    """Verify that SettingValue is a proper oneof: setting one variant leaves all others unset."""

    @pytest.mark.parametrize(
        "sv, expected_oneof",
        [
            (settings_definition_pb2.SettingValue(string_value="gpu"), "string_value"),
            (settings_definition_pb2.SettingValue(int_value=42), "int_value"),
            (settings_definition_pb2.SettingValue(bool_value=True), "bool_value"),
            (
                settings_definition_pb2.SettingValue(
                    list_value=settings_definition_pb2.StringValues(values=["a", "b"])
                ),
                "list_value",
            ),
            (
                settings_definition_pb2.SettingValue(
                    map_value=settings_definition_pb2.StringMap(entries={"k": "v"})
                ),
                "map_value",
            ),
        ],
        ids=["string", "int", "bool", "list", "map"],
    )
    def test_only_one_variant_set(self, sv, expected_oneof):
        all_variants = {"string_value", "int_value", "bool_value", "list_value", "map_value", "state"}
        assert sv.WhichOneof("value") == expected_oneof
        for variant in all_variants - {expected_oneof}:
            assert sv.WhichOneof("value") != variant


class TestSettingsWireFormat:
    """Verify Settings proto survives marshal (SerializeToString) → unmarshal (FromString) roundtrip."""

    def test_full_settings_roundtrip(self):
        original = settings_definition_pb2.Settings(
            run=settings_definition_pb2.RunSettings(
                default_queue=settings_definition_pb2.SettingValue(string_value="gpu"),
                run_concurrency=settings_definition_pb2.SettingValue(int_value=10),
                action_concurrency=settings_definition_pb2.SettingValue(int_value=5),
            ),
            security=settings_definition_pb2.SecuritySettings(
                service_account=settings_definition_pb2.SettingValue(string_value="ml-sa"),
            ),
            storage=settings_definition_pb2.StorageSettings(
                raw_data_path=settings_definition_pb2.SettingValue(string_value="s3://bucket/data"),
            ),
            task_resource=settings_definition_pb2.TaskResourceSettings(
                defaults=settings_definition_pb2.TaskResourceDefaults(
                    min_cpu=settings_definition_pb2.SettingValue(string_value="2"),
                    max_memory=settings_definition_pb2.SettingValue(string_value="8Gi"),
                ),
                mirror_limits_request=settings_definition_pb2.SettingValue(bool_value=True),
            ),
            labels=settings_definition_pb2.SettingValue(
                map_value=settings_definition_pb2.StringMap(entries={"env": "prod", "team": "ml"})
            ),
            annotations=settings_definition_pb2.SettingValue(
                map_value=settings_definition_pb2.StringMap(entries={"oncall": "ml-team"})
            ),
            environment_variables=settings_definition_pb2.SettingValue(
                map_value=settings_definition_pb2.StringMap(entries={"DEBUG": "0"})
            ),
        )

        wire = original.SerializeToString()
        restored = settings_definition_pb2.Settings.FromString(wire)

        assert restored == original
        # Spot-check individual fields survived the roundtrip
        assert restored.run.default_queue.string_value == "gpu"
        assert restored.run.run_concurrency.int_value == 10
        assert restored.security.service_account.string_value == "ml-sa"
        assert restored.task_resource.defaults.min_cpu.string_value == "2"
        assert restored.task_resource.mirror_limits_request.bool_value is True
        assert dict(restored.labels.map_value.entries) == {"env": "prod", "team": "ml"}

        # Verify oneof exclusivity after roundtrip: run_concurrency is int_value,
        # so all other variants must be unset.
        rc = restored.run.run_concurrency
        assert rc.WhichOneof("value") == "int_value"
        assert rc.string_value == ""
        assert rc.bool_value is False
        assert not rc.HasField("list_value")
        assert not rc.HasField("map_value")
        assert not rc.HasField("state")

    def test_setting_value_each_variant_roundtrip(self):
        variants = [
            settings_definition_pb2.SettingValue(string_value="hello"),
            settings_definition_pb2.SettingValue(int_value=99),
            settings_definition_pb2.SettingValue(bool_value=False),
            settings_definition_pb2.SettingValue(
                list_value=settings_definition_pb2.StringValues(values=["x", "y", "z"])
            ),
            settings_definition_pb2.SettingValue(
                map_value=settings_definition_pb2.StringMap(entries={"a": "1", "b": "2"})
            ),
            settings_definition_pb2.SettingValue(state=settings_definition_pb2.SETTING_STATE_INHERIT),
        ]
        for original in variants:
            wire = original.SerializeToString()
            restored = settings_definition_pb2.SettingValue.FromString(wire)
            assert restored == original
            assert restored.WhichOneof("value") == original.WhichOneof("value")

    def test_raw_data_survives_wire_roundtrip(self):
        """flat dict → RawSettingsRecord → wire → restore → _raw_data_to_flat should be identity."""
        overrides = {
            "run.default_queue": "gpu",
            "run.run_concurrency": 10,
            "security.service_account": "my-sa",
            "task_resource.defaults.min_cpu": "2",
            "task_resource.mirror_limits_request": True,
            "labels": {"env": "prod"},
        }
        record = _make_raw_settings_record(org="myorg", version=1, **overrides)
        wire = record.SerializeToString()
        restored = settings_service_pb2.RawSettingsRecord.FromString(wire)
        flat = dict(_raw_data_to_flat(restored.data))
        assert flat == overrides

    def test_raw_settings_record_roundtrip(self):
        """RawSettingsRecord (used in GetSettingsForEditRawResponse) survives wire roundtrip."""
        record = _make_raw_settings_record(
            org="myorg",
            domain="prod",
            project="ml",
            version=42,
            **{"run.default_queue": "gpu", "security.service_account": "sa"},
        )

        wire = record.SerializeToString()
        restored = settings_service_pb2.RawSettingsRecord.FromString(wire)

        assert restored.key.org == "myorg"
        assert restored.key.domain == "prod"
        assert restored.key.project == "ml"
        assert restored.version == 42
        assert restored.data["run.default_queue"].string_value == "gpu"
        assert restored.data["security.service_account"].string_value == "sa"

    def test_empty_raw_record_roundtrip(self):
        record = settings_service_pb2.RawSettingsRecord()
        wire = record.SerializeToString()
        restored = settings_service_pb2.RawSettingsRecord.FromString(wire)
        assert restored == record
        assert _raw_data_to_flat(restored.data) == []


class TestMakeSettingValue:
    def test_string(self):
        sv = _make_setting_value("gpu")
        assert sv.WhichOneof("value") == "string_value"
        assert sv.string_value == "gpu"

    def test_int(self):
        sv = _make_setting_value(42)
        assert sv.WhichOneof("value") == "int_value"
        assert sv.int_value == 42

    def test_bool(self):
        sv = _make_setting_value(True)
        assert sv.WhichOneof("value") == "bool_value"
        assert sv.bool_value is True

    def test_list(self):
        sv = _make_setting_value(["a", "b"])
        assert sv.WhichOneof("value") == "list_value"
        assert list(sv.list_value.values) == ["a", "b"]

    def test_dict(self):
        sv = _make_setting_value({"k": "v"})
        assert sv.WhichOneof("value") == "map_value"
        assert dict(sv.map_value.entries) == {"k": "v"}

    def test_other_type_coerced_to_string(self):
        sv = _make_setting_value(3.14)
        assert sv.WhichOneof("value") == "string_value"
        assert sv.string_value == "3.14"


class TestRawDataToFlat:
    def test_extracts_values(self):
        data = {
            "run.default_queue": settings_definition_pb2.SettingValue(string_value="gpu"),
            "run.run_concurrency": settings_definition_pb2.SettingValue(int_value=10),
        }
        result = dict(_raw_data_to_flat(data))
        assert result["run.default_queue"] == "gpu"
        assert result["run.run_concurrency"] == 10

    def test_all_value_types(self):
        data = {
            "str_key": settings_definition_pb2.SettingValue(string_value="hello"),
            "int_key": settings_definition_pb2.SettingValue(int_value=42),
            "bool_key": settings_definition_pb2.SettingValue(bool_value=True),
            "list_key": settings_definition_pb2.SettingValue(
                list_value=settings_definition_pb2.StringValues(values=["a", "b"])
            ),
            "map_key": settings_definition_pb2.SettingValue(
                map_value=settings_definition_pb2.StringMap(entries={"k": "v"})
            ),
        }
        result = dict(_raw_data_to_flat(data))
        assert result["str_key"] == "hello"
        assert result["int_key"] == 42
        assert result["bool_key"] is True
        assert result["list_key"] == ["a", "b"]
        assert result["map_key"] == {"k": "v"}

    def test_skips_inherited_values(self):
        data = {
            "run.default_queue": settings_definition_pb2.SettingValue(string_value="gpu"),
            "run.run_concurrency": settings_definition_pb2.SettingValue(
                state=settings_definition_pb2.SETTING_STATE_INHERIT
            ),
        }
        result = dict(_raw_data_to_flat(data))
        assert "run.default_queue" in result
        assert "run.run_concurrency" not in result

    def test_empty_data(self):
        assert _raw_data_to_flat({}) == []

    def test_arbitrary_keys(self):
        data = {
            "custom.whatever.key": settings_definition_pb2.SettingValue(string_value="val"),
        }
        result = dict(_raw_data_to_flat(data))
        assert result["custom.whatever.key"] == "val"


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
    def test_local_and_inherited(self):
        settings = Settings(
            effective_settings=[
                EffectiveSetting(key="run.default_queue", value="gpu", origin=SettingOrigin("PROJECT", "prod", "ml")),
                EffectiveSetting(key="run.run_concurrency", value=10, origin=SettingOrigin("DOMAIN", "prod")),
                EffectiveSetting(key="security.service_account", value="sa", origin=SettingOrigin("ORG")),
            ],
            local_settings=[
                LocalSetting(key="run.default_queue", value="gpu"),
            ],
        )
        yaml = settings.to_yaml()
        assert "run.default_queue: gpu" in yaml
        # Inherited should be commented
        assert "# run.run_concurrency: 10" in yaml
        assert "# security.service_account: sa" in yaml
        assert "inherited from DOMAIN(prod)" in yaml
        assert "inherited from ORG" in yaml

    def test_no_local_settings(self):
        settings = Settings(
            effective_settings=[
                EffectiveSetting(key="run.default_queue", value="cpu", origin=SettingOrigin("ORG")),
            ],
            local_settings=[],
        )
        yaml = settings.to_yaml()
        assert "# Local overrides" not in yaml
        assert "# run.default_queue: cpu" in yaml

    def test_empty(self):
        settings = Settings(effective_settings=[], local_settings=[])
        yaml = settings.to_yaml()
        assert yaml == ""


class TestParseYaml:
    def test_parse_overrides(self):
        yaml_content = """# Local overrides
run.default_queue: gpu
run.run_concurrency: 10
security.service_account: my-sa

# Inherited settings (uncomment to override)
# storage.raw_data_path: s3://bucket  # inherited from ORG
"""
        overrides = Settings.parse_yaml(yaml_content)
        assert overrides == {
            "run.default_queue": "gpu",
            "run.run_concurrency": 10,
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


# ---------------------------------------------------------------------------
# Settings.get and Settings.update (gRPC integration)
# ---------------------------------------------------------------------------


def _make_raw_settings_record(
    org: str = "",
    domain: str = "",
    project: str = "",
    version: int = 1,
    **setting_values,
) -> settings_service_pb2.RawSettingsRecord:
    """Build a RawSettingsRecord with the given flat key-value settings."""
    key = settings_definition_pb2.SettingsKey(org=org, domain=domain, project=project)
    record = settings_service_pb2.RawSettingsRecord(key=key, version=version)
    for k, v in setting_values.items():
        record.data[k].CopyFrom(_make_setting_value(v))
    return record


@pytest.fixture
def mock_settings_service():
    svc = MagicMock()
    svc.GetSettingsForEditRaw = AsyncMock()
    svc.UpdateSettingsRaw = AsyncMock(return_value=settings_service_pb2.UpdateSettingsRawResponse())
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
        mock_settings_service.GetSettingsForEditRaw.return_value = settings_service_pb2.GetSettingsForEditRawResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg"),
            levels=[
                _make_raw_settings_record(
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
            settings = Settings.get()

        assert len(settings.effective_settings) == 2
        assert len(settings.local_settings) == 2
        assert settings._version == 5
        assert settings.domain is None
        assert settings.project is None

        effective_keys = {s.key for s in settings.effective_settings}
        assert "run.default_queue" in effective_keys
        assert "security.service_account" in effective_keys

    def test_get_project_scope_with_inheritance(self, mock_client, mock_settings_service, mock_init_config):
        org_record = _make_raw_settings_record(
            org="myorg",
            version=1,
            **{"run.default_queue": "cpu", "security.service_account": "org-sa"},
        )
        domain_record = _make_raw_settings_record(
            org="myorg",
            domain="prod",
            version=3,
            **{"run.default_queue": "gpu"},
        )
        project_record = _make_raw_settings_record(
            org="myorg",
            domain="prod",
            project="ml",
            version=7,
            **{"security.service_account": "ml-sa"},
        )

        mock_settings_service.GetSettingsForEditRaw.return_value = (
            settings_service_pb2.GetSettingsForEditRawResponse(
                requestedKey=settings_definition_pb2.SettingsKey(org="myorg", domain="prod", project="ml"),
                levels=[org_record, domain_record, project_record],
            )
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get(domain="prod", project="ml")

        assert settings._version == 7
        assert settings.domain == "prod"
        assert settings.project == "ml"

        # Effective settings should have overridden values
        effective = {s.key: s for s in settings.effective_settings}
        assert effective["run.default_queue"].value == "gpu"
        assert effective["run.default_queue"].origin.scope_type == "DOMAIN"
        assert effective["security.service_account"].value == "ml-sa"
        assert effective["security.service_account"].origin.scope_type == "PROJECT"

        # Local settings should only be from the most specific (project) level
        local_keys = {s.key for s in settings.local_settings}
        assert local_keys == {"security.service_account"}

    def test_get_sends_correct_key(self, mock_client, mock_settings_service, mock_init_config):
        mock_settings_service.GetSettingsForEditRaw.return_value = (
            settings_service_pb2.GetSettingsForEditRawResponse(levels=[])
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            Settings.get(domain="staging", project="web")

        req = mock_settings_service.GetSettingsForEditRaw.call_args[0][0]
        assert req.key.org == "myorg"
        assert req.key.domain == "staging"
        assert req.key.project == "web"

    def test_get_empty_levels(self, mock_client, mock_settings_service, mock_init_config):
        mock_settings_service.GetSettingsForEditRaw.return_value = (
            settings_service_pb2.GetSettingsForEditRawResponse(levels=[])
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings = Settings.get()

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
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings.update({"run.default_queue": "gpu", "run.run_concurrency": 5})

        mock_settings_service.UpdateSettingsRaw.assert_called_once()
        req = mock_settings_service.UpdateSettingsRaw.call_args[0][0]
        assert req.key.org == "myorg"
        assert req.key.domain == "prod"
        assert req.key.project == "ml"
        assert req.version == 7
        assert req.data["run.default_queue"].string_value == "gpu"
        assert req.data["run.run_concurrency"].int_value == 5

    def test_update_refreshes_local_state(self, mock_client, mock_settings_service, mock_init_config):
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="old.key", value="old")],
            domain="prod",
            _version=3,
        )

        with (
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings.update({"run.default_queue": "gpu"})

        assert len(settings.local_settings) == 1
        assert settings.local_settings[0].key == "run.default_queue"
        assert settings.local_settings[0].value == "gpu"

    def test_update_empty_overrides(self, mock_client, mock_settings_service, mock_init_config):
        settings = Settings(
            effective_settings=[],
            local_settings=[LocalSetting(key="run.default_queue", value="gpu")],
            _version=1,
        )

        with (
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            settings.update({})

        req = mock_settings_service.UpdateSettingsRaw.call_args[0][0]
        assert len(req.data) == 0
        assert settings.local_settings == []


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
