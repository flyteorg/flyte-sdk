"""Profile selection in the config reader."""

from __future__ import annotations

import pathlib

import pytest

import flyte.config as config
from flyte.config._reader import ConfigFile, ProfileNotFoundError

_CONFIG = """
admin:
  endpoint: dns:///default.example.com
  insecure: true
task:
  org: shared-org
  project: shared
  domain: development
image:
  builder: local
profiles:
  prod:
    admin:
      endpoint: dns:///prod.example.com
      insecure: false
    task:
      domain: production
  gpu:
    admin:
      endpoint: dns:///gpu.example.com
"""


@pytest.fixture
def cfg_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "config.yaml"
    p.write_text(_CONFIG)
    return p


def test_no_profile_reads_top_level(cfg_file: pathlib.Path) -> None:
    cfg = config.auto(cfg_file)
    assert cfg.platform.endpoint == "dns:///default.example.com"
    assert cfg.task.domain == "development"
    assert cfg.profile is None


def test_profile_overrides_and_inherits(cfg_file: pathlib.Path) -> None:
    cfg = config.auto(cfg_file, profile="prod")
    # Overridden by the profile.
    assert cfg.platform.endpoint == "dns:///prod.example.com"
    assert cfg.task.domain == "production"
    # Not set by the profile, so inherited from the top level.
    assert cfg.task.project == "shared"
    assert cfg.task.org == "shared-org"
    assert cfg.image.builder == "local"
    assert cfg.profile == "prod"


def test_profile_only_overrides_what_it_sets(cfg_file: pathlib.Path) -> None:
    cfg = config.auto(cfg_file, profile="gpu")
    assert cfg.platform.endpoint == "dns:///gpu.example.com"
    assert cfg.task.domain == "development"


def test_unknown_profile_raises(cfg_file: pathlib.Path) -> None:
    with pytest.raises(ProfileNotFoundError, match="nope"):
        config.auto(cfg_file, profile="nope")


def test_list_profiles(cfg_file: pathlib.Path) -> None:
    assert sorted(config.list_profiles(cfg_file)) == ["gpu", "prod"]


def test_list_profiles_on_file_without_profiles(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("admin:\n  endpoint: dns:///only.example.com\n")
    assert config.list_profiles(p) == []


def test_file_without_profiles_ignores_active_profile(tmp_path: pathlib.Path) -> None:
    """A file that declares no profiles is not silently mis-read -- asking for one is an error."""
    p = tmp_path / "config.yaml"
    p.write_text("admin:\n  endpoint: dns:///only.example.com\n")
    with pytest.raises(ProfileNotFoundError):
        config.auto(p, profile="prod")


def test_profiles_are_cached_separately(cfg_file: pathlib.Path) -> None:
    """Regression: the ConfigFile cache is keyed by path *and* profile.

    Sharing one entry across profiles would make the second read return the first profile's
    file object, silently sending work to the wrong control plane.
    """
    a = config.auto(cfg_file, profile="prod")
    b = config.auto(cfg_file, profile="gpu")
    c = config.auto(cfg_file, profile="prod")
    assert a.platform.endpoint == c.platform.endpoint == "dns:///prod.example.com"
    assert b.platform.endpoint == "dns:///gpu.example.com"


def test_active_profile_env_var(cfg_file: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLYTE_PROFILE", "prod")
    assert config.get_active_profile() == "prod"
    assert config.auto(cfg_file).platform.endpoint == "dns:///prod.example.com"


def test_set_active_profile_beats_env_var(cfg_file: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLYTE_PROFILE", "prod")
    try:
        config.set_active_profile("gpu")
        assert config.get_active_profile() == "gpu"
        assert config.auto(cfg_file).platform.endpoint == "dns:///gpu.example.com"
    finally:
        config.set_active_profile(None)


def test_explicit_profile_beats_active_profile(cfg_file: pathlib.Path) -> None:
    try:
        config.set_active_profile("gpu")
        assert config.auto(cfg_file, profile="prod").platform.endpoint == "dns:///prod.example.com"
    finally:
        config.set_active_profile(None)


def test_env_var_still_wins_over_profile(cfg_file: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile selects the file section; an explicit FLYTE_* env var still overrides it.

    This mirrors AWS, where AWS_PROFILE picks the profile but explicit env vars still win.
    """
    monkeypatch.setenv("FLYTE_ADMIN_ENDPOINT", "dns:///override.example.com")
    assert config.auto(cfg_file, profile="prod").platform.endpoint == "dns:///override.example.com"


def test_configfile_profiles_property(cfg_file: pathlib.Path) -> None:
    f = ConfigFile(str(cfg_file))
    assert sorted(f.profiles) == ["gpu", "prod"]
    assert f.profile is None
    assert ConfigFile(str(cfg_file), profile="prod").profile == "prod"


def test_already_loaded_configfile_keeps_its_profile(cfg_file: pathlib.Path) -> None:
    """An already-built ConfigFile is returned as-is; its profile was fixed at construction."""
    f = ConfigFile(str(cfg_file), profile="prod")
    assert config.get_config_file(f, profile="gpu") is f
    assert config.auto(f).platform.endpoint == "dns:///prod.example.com"
