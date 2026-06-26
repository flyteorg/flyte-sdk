# flyteplugins-remote-tui

Terminal UI for remote Flyte v2 clusters. Mirrors the Flyte 2 web UI (runs, actions, logs, tasks, apps, triggers) using the same Textual style as `flyte run --local --tui`.

## Install

The plugin must be installed in the **same Python environment as the `flyte` CLI**.
Entry points alone are not enough if `flyte` and the plugin live in different envs
(for example `uv tool install flyte` vs a project virtualenv).

```bash
pip install flyteplugins-remote-tui
# or from this repo:
pip install -e plugins/remote_tui
```

Requires a configured Flyte endpoint (`flyte create config --endpoint ...`).

## Usage

```bash
flyte start remote-tui
flyte start remote-tui --config /path/to/config.yaml
```

See [REMOTE_TUI_PLUGIN.md](./REMOTE_TUI_PLUGIN.md) for the full product spec.

## Navigation (Devbox UI layout)

1. **Projects** — list all projects; `enter` opens a project.
2. **Project workspace** — sidebar: Runs, Triggers, Tasks, Apps; `enter` opens detail.
3. **Run detail** — action tree, inputs/outputs, logs.

## Keyboard

| Key | Action |
|-----|--------|
| `enter` | Open project / open detail |
| `escape` | Back (detail → project, project → projects) |
| `1`–`4` | Runs / Triggers / Tasks / Apps (inside a project) |
| `r` | Refresh list |
| `d` / `l` | Details / Logs (run detail) |
| `a` | Abort run |
| `q` | Quit |
