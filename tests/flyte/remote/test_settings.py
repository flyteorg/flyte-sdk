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
    _flat_overrides_to_settings_proto,
    _make_setting_value,
    _scope_origin_from_key,
    _settings_proto_to_flat,
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


class TestSettingsProtoToFlat:
    def test_simple_run_settings(self):
        proto = settings_definition_pb2.Settings(
            run=settings_definition_pb2.RunSettings(
                default_queue=settings_definition_pb2.SettingValue(string_value="gpu"),
                run_concurrency=settings_definition_pb2.SettingValue(int_value=10),
            )
        )
        result = dict(_settings_proto_to_flat(proto))
        assert result["run.default_queue"] == "gpu"
        assert result["run.run_concurrency"] == 10

    def test_nested_task_resource(self):
        proto = settings_definition_pb2.Settings(
            task_resource=settings_definition_pb2.TaskResourceSettings(
                defaults=settings_definition_pb2.TaskResourceDefaults(
                    min_cpu=settings_definition_pb2.SettingValue(string_value="2"),
                    max_memory=settings_definition_pb2.SettingValue(string_value="8Gi"),
                )
            )
        )
        result = dict(_settings_proto_to_flat(proto))
        assert result["task_resource.defaults.min_cpu"] == "2"
        assert result["task_resource.defaults.max_memory"] == "8Gi"

    def test_top_level_setting_value(self):
        proto = settings_definition_pb2.Settings(
            labels=settings_definition_pb2.SettingValue(
                map_value=settings_definition_pb2.StringMap(entries={"env": "prod"})
            )
        )
        result = dict(_settings_proto_to_flat(proto))
        assert result["labels"] == {"env": "prod"}

    def test_skips_inherited_values(self):
        proto = settings_definition_pb2.Settings(
            run=settings_definition_pb2.RunSettings(
                default_queue=settings_definition_pb2.SettingValue(string_value="gpu"),
                run_concurrency=settings_definition_pb2.SettingValue(
                    state=settings_definition_pb2.SETTING_STATE_INHERIT
                ),
            )
        )
        result = dict(_settings_proto_to_flat(proto))
        assert "run.default_queue" in result
        assert "run.run_concurrency" not in result

    def test_empty_settings(self):
        proto = settings_definition_pb2.Settings()
        result = _settings_proto_to_flat(proto)
        assert result == []


class TestFlatOverridesToSettingsProto:
    def test_two_part_keys(self):
        proto = _flat_overrides_to_settings_proto(
            {
                "run.default_queue": "gpu",
                "security.service_account": "my-sa",
            }
        )
        assert proto.run.default_queue.string_value == "gpu"
        assert proto.security.service_account.string_value == "my-sa"

    def test_three_part_keys(self):
        proto = _flat_overrides_to_settings_proto(
            {
                "task_resource.defaults.min_cpu": "2",
                "task_resource.defaults.max_memory": "8Gi",
            }
        )
        assert proto.task_resource.defaults.min_cpu.string_value == "2"
        assert proto.task_resource.defaults.max_memory.string_value == "8Gi"

    def test_top_level_keys(self):
        proto = _flat_overrides_to_settings_proto(
            {
                "labels": {"env": "prod"},
            }
        )
        assert dict(proto.labels.map_value.entries) == {"env": "prod"}

    def test_int_value(self):
        proto = _flat_overrides_to_settings_proto(
            {
                "run.run_concurrency": 10,
            }
        )
        assert proto.run.run_concurrency.int_value == 10

    def test_bool_value(self):
        proto = _flat_overrides_to_settings_proto(
            {
                "task_resource.mirror_limits_request": True,
            }
        )
        assert proto.task_resource.mirror_limits_request.bool_value is True

    def test_empty_overrides(self):
        proto = _flat_overrides_to_settings_proto({})
        assert proto == settings_definition_pb2.Settings()

    def test_roundtrip(self):
        """Flatten → unflatten should preserve values."""
        original = {
            "run.default_queue": "gpu",
            "run.run_concurrency": 10,
            "security.service_account": "my-sa",
            "task_resource.defaults.min_cpu": "2",
            "labels": {"env": "prod"},
        }
        proto = _flat_overrides_to_settings_proto(original)
        flat = dict(_settings_proto_to_flat(proto))
        assert flat == original


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


def _make_settings_record(
    org: str = "",
    domain: str = "",
    project: str = "",
    version: int = 1,
    **setting_values,
) -> settings_service_pb2.SettingsRecord:
    """Build a SettingsRecord with the given flat key-value settings."""
    key = settings_definition_pb2.SettingsKey(org=org, domain=domain, project=project)
    settings_proto = _flat_overrides_to_settings_proto(setting_values)
    return settings_service_pb2.SettingsRecord(key=key, settings=settings_proto, version=version)


@pytest.fixture
def mock_settings_service():
    svc = MagicMock()
    svc.GetSettingsForEdit = AsyncMock()
    svc.UpdateSettings = AsyncMock(return_value=settings_service_pb2.UpdateSettingsResponse())
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
        mock_settings_service.GetSettingsForEdit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg"),
            levels=[
                _make_settings_record(
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
        org_record = _make_settings_record(
            org="myorg",
            version=1,
            **{"run.default_queue": "cpu", "security.service_account": "org-sa"},
        )
        domain_record = _make_settings_record(
            org="myorg",
            domain="prod",
            version=3,
            **{"run.default_queue": "gpu"},
        )
        project_record = _make_settings_record(
            org="myorg",
            domain="prod",
            project="ml",
            version=7,
            **{"security.service_account": "ml-sa"},
        )

        mock_settings_service.GetSettingsForEdit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            requestedKey=settings_definition_pb2.SettingsKey(org="myorg", domain="prod", project="ml"),
            levels=[org_record, domain_record, project_record],
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
        mock_settings_service.GetSettingsForEdit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            levels=[]
        )

        with (
            patch(_PATCH_ENSURE, new_callable=AsyncMock),
            patch(_PATCH_CLIENT, return_value=mock_client),
            patch(_PATCH_CONFIG, return_value=mock_init_config),
        ):
            Settings.get(domain="staging", project="web")

        req = mock_settings_service.GetSettingsForEdit.call_args[0][0]
        assert req.key.org == "myorg"
        assert req.key.domain == "staging"
        assert req.key.project == "web"

    def test_get_empty_levels(self, mock_client, mock_settings_service, mock_init_config):
        mock_settings_service.GetSettingsForEdit.return_value = settings_service_pb2.GetSettingsForEditResponse(
            levels=[]
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

        mock_settings_service.UpdateSettings.assert_called_once()
        req = mock_settings_service.UpdateSettings.call_args[0][0]
        assert req.key.org == "myorg"
        assert req.key.domain == "prod"
        assert req.key.project == "ml"
        assert req.version == 7
        assert req.settings.run.default_queue.string_value == "gpu"
        assert req.settings.run.run_concurrency.int_value == 5

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

        req = mock_settings_service.UpdateSettings.call_args[0][0]
        assert req.settings == settings_definition_pb2.Settings()
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
