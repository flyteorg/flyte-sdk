# flyteplugins-remote-tui

Terminal UI for remote Flyte v2 clusters. Mirrors the Flyte 2 web UI (runs, actions, logs, tasks, apps, triggers) using the same Textual style as `flyte run --local --tui`.

## Install

```bash
pip install flyteplugins-remote-tui
# or from this repo:
pip install -e plugins/remote_tui
```

Requires a configured Flyte endpoint (`flyte create config --endpoint ...`).

## Usage

```bash
flyte start remote-tui
flyte start remote-tui --project my-project --domain development
```

See [REMOTE_TUI_PLUGIN.md](./REMOTE_TUI_PLUGIN.md) for the full product spec.

## Keyboard

| Key | Action |
|-----|--------|
| `1`–`4` | Runs / Tasks / Apps / Triggers |
| `r` | Refresh list |
| `enter` | Open detail |
| `escape` | Back |
| `d` / `l` | Details / Logs (run detail) |
| `a` | Abort run (when terminal confirmation applies) |
| `q` | Quit |
