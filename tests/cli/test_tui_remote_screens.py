"""Headless tests for the remote TUI screens.

These drive the real ``RemoteTUIApp`` through Textual's test pilot with the
remote client calls monkeypatched, verifying that data is fetched off the UI
thread and rendered into the tables (i.e. the app does not block and does load
items). No network/backend is required.
"""

from __future__ import annotations

import asyncio

import pytest

from flyte.cli._tui._remote import _app as app_mod
from flyte.cli._tui._remote import _screens as screens_mod
from flyte.cli._tui._remote._app import RemoteTUIApp
from flyte.cli._tui._remote._client import ClusterContext, PagedResult
from flyte.cli._tui._remote._screens import ProjectHubScreen, ProjectsScreen


class _FakePB2:
    def __init__(self, id: str, name: str) -> None:
        self.id = id
        self.name = name
        self.labels = None


class _FakeProject:
    def __init__(self, id: str, name: str) -> None:
        self.pb2 = _FakePB2(id, name)


class _FakeApp:
    def __init__(self, name: str) -> None:
        self.name = name
        self.deployment_status = "DEPLOYMENT_STATUS_RUNNING"
        self.endpoint = f"https://{name}.example.org"


async def _wait_until(pilot, cond, tries: int = 200) -> bool:
    for _ in range(tries):
        try:
            if cond():
                return True
        except Exception:
            # Widgets may not be mounted yet while background workers run.
            pass
        await pilot.pause()
    return False


@pytest.fixture
def patched_remote(monkeypatch):
    """Patch cluster init + list calls so the TUI runs without a backend."""
    monkeypatch.setattr(
        app_mod,
        "init_cluster",
        lambda config=None: ClusterContext(domain="development", org="demo", endpoint="dns:///x", default_project="p"),
    )
    monkeypatch.setattr(screens_mod, "get_recent_projects", lambda config_key: [])
    monkeypatch.setattr(screens_mod, "record_recent_project", lambda *a, **k: None)
    monkeypatch.setattr(screens_mod, "activate_project", lambda **k: None)


def test_projects_screen_loads_items(patched_remote, monkeypatch):
    projects = [_FakeProject("proj-a", "Project A"), _FakeProject("proj-b", "Project B")]
    monkeypatch.setattr(screens_mod, "list_projects", lambda **k: projects)

    async def run() -> None:
        app = RemoteTUIApp(config=None)
        async with app.run_test(size=(120, 40)) as pilot:
            # The connect step + project fetch run in background threads; the app
            # should reach the ProjectsScreen and render both rows.
            ok = await _wait_until(
                pilot,
                lambda: (
                    isinstance(app.screen, ProjectsScreen) and app.screen.query_one("#projects-table").row_count == 2
                ),
            )
            assert ok, "projects table did not populate"
            table = app.screen.query_one("#projects-table")
            first_cell = table.get_cell_at((0, 0))
            assert "Loading" not in str(first_cell)

    asyncio.run(run())


def test_projects_screen_shows_error_row_on_failure(patched_remote, monkeypatch):
    def _boom(**k):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(screens_mod, "list_projects", _boom)

    async def run() -> None:
        app = RemoteTUIApp(config=None)
        async with app.run_test(size=(120, 40)) as pilot:
            ok = await _wait_until(
                pilot,
                lambda: (
                    isinstance(app.screen, ProjectsScreen)
                    and app.screen.query_one("#projects-table").row_count == 1
                    and "Error" in str(app.screen.query_one("#projects-table").get_cell_at((0, 0)))
                ),
            )
            assert ok, "error row not shown"

    asyncio.run(run())


def test_project_hub_loads_section(patched_remote, monkeypatch):
    projects = [_FakeProject("proj-a", "Project A")]
    monkeypatch.setattr(screens_mod, "list_projects", lambda **k: projects)
    # Default hub section is "runs"; return an empty page so we exercise the
    # fetch->render worker path without building full fake Run protos.
    empty = PagedResult(items=[], page=0, page_size=10, has_next=False)
    monkeypatch.setattr(screens_mod, "list_runs_paginated", lambda **k: empty)
    monkeypatch.setattr(screens_mod, "list_apps_paginated", lambda **k: PagedResult([_FakeApp("svc")], 0, 10, False))

    async def run() -> None:
        app = RemoteTUIApp(config=None)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _wait_until(
                pilot,
                lambda: (
                    isinstance(app.screen, ProjectsScreen) and app.screen.query_one("#projects-table").row_count == 1
                ),
            )
            # Open the project -> pushes ProjectHubScreen and loads the runs section.
            app.screen.query_one("#projects-table").focus()
            await pilot.press("enter")
            assert await _wait_until(pilot, lambda: isinstance(app.screen, ProjectHubScreen))
            # Runs section loads (empty) without error.
            assert await _wait_until(pilot, lambda: app.screen.sub_title == "0 on page"), (
                f"hub did not finish loading: sub_title={app.screen.sub_title!r}"
            )
            # Switch to the apps section and confirm the fake app row renders.
            hub = app.screen
            hub.action_show_apps()
            assert await _wait_until(
                pilot,
                lambda: (
                    hub.query_one("#hub-table").row_count == 1
                    and "svc" in str(hub.query_one("#hub-table").get_cell_at((0, 0)))
                ),
            )

    asyncio.run(run())


def test_entity_detail_loads_off_thread(monkeypatch):
    import flyte.remote

    class _FakeAppEntity:
        name = "svc"
        deployment_status = "DEPLOYMENT_STATUS_RUNNING"
        endpoint = "https://svc.example.org"
        url = "https://console.example.org/svc"

    monkeypatch.setattr(flyte.remote.App, "get", classmethod(lambda cls, *a, **k: _FakeAppEntity()))

    from textual.app import App
    from textual.widgets import Static

    from flyte.cli._tui._remote._screens import EntityDetailScreen

    async def run() -> None:
        class _Host(App):
            async def on_mount(self) -> None:
                await self.push_screen(EntityDetailScreen("App", "svc"))

        app = _Host()
        async with app.run_test(size=(100, 30)) as pilot:
            ok = await _wait_until(
                pilot,
                lambda: "svc.example.org" in str(app.screen.query_one("#detail-body", Static).render()),
            )
            assert ok, "entity detail did not load"

    asyncio.run(run())
