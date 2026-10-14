# flyteplugins-redis

Redis-backed metadata storage for flyte, selected purely by path: any `redis://` path used with
`flyte.storage` (and therefore any run whose metadata prefix is a `redis://` path) reads and writes
Redis instead of an object store.

## Design

flyte's metadata IO (`inputs.pb`, `outputs.pb`, `error.pb`) flows exclusively through
`flyte.storage` (`put_stream` / `get_stream` / `exists` / `get` / `put`), which resolves an fsspec
filesystem from the path's protocol. This plugin provides `RedisFileSystem`, an fsspec
`AsyncFileSystem` that maps file semantics onto Redis string keys, and registers it for the
`redis` protocol through fsspec's `fsspec.specs` entry point.

Registration is lazy: installing this package only records the class path with fsspec. Nothing is
imported — and a Redis server is never contacted — until a `redis://` path is actually resolved.
No changes to the flyte core are needed.

### Path model

```
redis://[user:password@]host[:port]/key/path
```

- The netloc selects the Redis server (logical db 0).
- Everything after the netloc is the key, verbatim: `redis://localhost:6379/flyte/runs/r1/inputs.pb`
  stores the key `flyte/runs/r1/inputs.pb`.
- One "file" = one Redis string value. Directories are emulated as key prefixes: a directory
  exists iff at least one key lives under it (`SCAN` with a `prefix/*` match).

## Usage

```bash
pip install flyteplugins-redis
```

Then point metadata at Redis. For hybrid runs:

```python
import flyte

flyte.init_from_config()
run = flyte.with_runcontext(
    mode="hybrid",
    run_base_dir="redis://localhost:6379/flyte/metadata",
).run(my_task, x=1)
```

Or use it directly with the storage API:

```python
import flyte.storage as storage

await storage.put_stream(b"hello", to_path="redis://localhost:6379/scratch/greeting")
data = b"".join([c async for c in storage.get_stream("redis://localhost:6379/scratch/greeting")])
```

## Limitations / notes

- **Metadata, not raw data.** A Redis string is capped at 512 MiB and lives in RAM. Metadata
  documents are tiny; keep large raw data (Files, Dirs, DataFrames) in an object store. Note that
  in hybrid mode `raw_data_path` is derived under `run_base_dir`, so offloaded raw data would also
  land in Redis — fine for experimentation, not for large datasets.
- Logical db is fixed at 0; authentication uses the standard `redis://user:password@host` URL form.
- No TTL is applied; keys persist until removed (`storage` deletes, `redis-cli DEL`, or server
  eviction policy).
- `rediss://` (TLS) is not wired up yet; it would only need a second entry point alias.

## Testing

Tests run against [fakeredis](https://github.com/cunla/fakeredis-py) — no server needed:

```bash
cd plugins/redis
uv run pytest tests/
```
