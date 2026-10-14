import importlib.metadata

from flyte._utils import entry_points as ep_module


def test_installed_distributions_are_scanned_once(monkeypatch):
    calls = []
    real = importlib.metadata.entry_points

    def counting(**params):
        calls.append(params)
        return real(**params)

    monkeypatch.setattr(importlib.metadata, "entry_points", counting)
    ep_module._installed_entry_points.cache_clear()
    try:
        first = ep_module.entry_points(group="console_scripts")
        ep_module.entry_points(group="flyte.plugins.types")
    finally:
        ep_module._installed_entry_points.cache_clear()

    assert calls == [{}]
    assert all(ep.group == "console_scripts" for ep in first)
