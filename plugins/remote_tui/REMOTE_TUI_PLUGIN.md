# Remote TUI Plugin — Product Specification

## Overview

The **Remote TUI** (`flyteplugins-remote-tui`) is a Textual-based terminal UI that connects to a remote Flyte v2 cluster via `flyte.remote` and mirrors the primary workflows of the Flyte 2 web UI ([flyte2-ui](https://github.com/unionai-oss/flyte2-ui)). It reuses the visual language and interaction patterns of the core library TUI (`flyte.cli._tui`) used by `flyte run --local --tui` and `flyte start tui`.

## Goals

1. **Parity with Flyte 2 UI** — Cover runs, actions (execution graph), inputs/outputs, logs, tasks, apps, and triggers.
2. **Remote-first** — All data comes from the control plane through `flyte.remote` (same surface as `flyte get` / SDK).
3. **Consistent UX** — Flyte purple theme, vim-style navigation (`j`/`k`), split-pane run detail (action tree + details/logs tabs).
4. **Plugin packaging** — Installable optional extra; registers `flyte start remote-tui` via `flyte.plugins.cli.commands`.

## Non-goals (v1)

- Launching new runs from the TUI (use `flyte run`).
- Full settings editor (use `flyte edit settings`).
- Secrets management UI.
- Graph visualization beyond a navigable tree (no canvas DAG in terminal).
- Authentication flows inside the TUI (assumes `flyte create config` / env is configured).

## References

| Source | Use |
|--------|-----|
| `src/flyte/cli/_tui` | Widgets, styling, `ActionTracker`, explore/run-detail layouts |
| `src/flyte/remote` | `Run`, `Action`, `ActionDetails`, `Task`, `App`, `Trigger`, logs streaming |
| `src/flyte/cli/_get.py` | CLI list/filter patterns for runs, actions, apps, triggers |
| Flyte v2 devbox (`flyteorg/flyte`) | Local cluster for manual testing |
| flyte2-ui | Information architecture: sidebar sections, run detail tabs, list filters |

## Flyte 2 UI mapping

| Web UI area | Remote TUI screen | Primary API |
|-------------|-------------------|-------------|
| Projects list | **Projects** (top level) | `Project.listall` |
| Project workspace | **Project hub** (sidebar) | `activate_project` re-inits client |
| Runs list | **Runs** (in project) | `Run.listall(project=…)` |
| Run detail / graph | **Run detail** | `Action.listall`, tree from `metadata.parent` / `group` |
| Run inputs/outputs | Details tab | `ActionDetails.inputs` / `outputs` |
| Run/action logs | Logs tab | `Action.get_logs` / `Run.get_logs` |
| Tasks registry | **Tasks** | `Task.listall` |
| Task detail | **Task detail** | `Task.get` → `fetch()` |
| Apps | **Apps** | `App.listall` |
| App detail | **App detail** | `App` status, endpoint, URL |
| Triggers | **Triggers** | `Trigger.listall` |
| Domain context | Header subtitle | Config file (`--config` / default `~/.flyte/config.yaml`) |

## Architecture

```
flyte start remote-tui
    └── RemoteTUIApp (Textual)
            ├── Sidebar: Runs | Tasks | Apps | Triggers
            ├── RunsScreen → RunDetailScreen (live poll / watch)
            ├── TasksScreen → TaskDetailScreen
            ├── AppsScreen → AppDetailScreen
            └── TriggersScreen

RemoteSyncService
    ├── init_cluster / activate_project (config file)
    ├── list_projects → ProjectHubScreen
    └── RemoteRunLoader → ActionTracker (core TUI model)
```

### Action tree construction

1. `Action.listall(for_run_name=…)` returns a flat list.
2. Build adjacency using `metadata.parent` (parent action name) and `metadata.group` (synthetic group nodes, same as local TUI).
3. Map `ActionPhase` → `ActionStatus` for tree icons.
4. On terminal actions, fetch inputs/outputs via `ActionDetails.inputs()` / `outputs()` (async, in worker thread).

### Live updates

- **Run detail**: background worker polls every N seconds (default 2s) or uses `Run.watch` when the run is non-terminal.
- **Logs**: dedicated worker streams `get_logs` into `RichLog` (same as local TUI log capture).

## CLI

```bash
# Install
pip install flyteplugins-remote-tui
# or: pip install flyte[tui] && pip install -e plugins/remote_tui

# Launch (uses ~/.flyte/config.yaml)
flyte start remote-tui

# Use a specific config file
flyte start remote-tui --config /path/to/config.yaml

# Poll interval for live run view (seconds)
flyte start remote-tui --poll-interval 3
```

Entry point: `flyte.plugins.cli.commands` → `start.remote-tui = flyteplugins.remote_tui._cli:remote_tui_cmd`

## Keyboard shortcuts

| Key | Context | Action |
|-----|---------|--------|
| `q` | Global | Quit |
| `r` | List screens | Refresh |
| `enter` | Lists | Open detail |
| `escape` | Detail | Back |
| `d` / `l` | Run detail | Details / Logs tab |
| `[` / `]` | Run detail | Previous / next attempt |
| `j` / `k` | Tables, tree, selects | Down / up |
| `a` | Run detail (terminal) | Abort run |
| `escape` | Project hub | Back to Projects |
| `1`–`4` | Project hub | Runs / Triggers / Tasks / Apps |

## Dependencies

- `flyte` (remote client, CLI config)
- `textual>=0.80` (aligned with `flyte[tui]` extra)

## Testing

- Unit tests for action-tree building and phase mapping (no cluster required).
- Optional integration tests marked `@pytest.mark.integration` when `FLYTE_REMOTE_INTEGRATION=1`.

## Future enhancements

- Watch mode using gRPC `watch_action_details` for sub-second updates.
- Trigger activate/deactivate from TUI.
- App activate/stop controls.
- Run launch wizard (task picker + inputs).
- Copy console URL to clipboard.
