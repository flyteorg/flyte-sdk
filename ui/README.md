# Flyte Local UI (FastAPI + Next.js)

This UI mirrors the dark Flyte run view and shows task duration, input, and output for a local workflow. Runs are persisted locally in SQLite.

## Backend

From `ui/backend`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The API runs at `http://localhost:8000`.

Runs are stored in `ui/backend/local_runs.db`.

## Showing `flyte run --local` in the UI

When you run:

```bash
flyte run --local examples/basics/hello.py main --x_list "[1,2,3,3,3,3,3,3,3,3,3,3]"
```

the UI will pick up the run automatically and show each `fn` task.

If you are running the SDK from this repo, install it in editable mode so the CLI picks up the local changes:

```bash
pip install -e .
```

If you want to override the DB path, set:

```bash
export FLYTE_UI_DB_PATH=/absolute/path/to/local_runs.db
```

## Reports tab

Flyte reports are rendered from `report.html` and shown in the UI under the `Reports` tab for each task.
To generate a report, mark the task with `report=True` and log content with `flyte.report` before flushing.

## Frontend

From `ui/frontend`:

```bash
npm install
npm run dev
```

The UI runs at `http://localhost:3000`.

## Notes

- Update the input list in the UI and hit `Rerun` to launch a new local run.
- The UI polls the backend while a run is in progress.
- Set `NEXT_PUBLIC_API_BASE` if your backend runs on a different host or port.

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```
