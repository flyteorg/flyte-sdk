"""
Devbox-local Apple Silicon accelerator, requested with `flyte.Resources(gpu="AppleSilicon")`.

A devbox container is a Linux VM with no access to the Mac's Metal GPU, so an Apple Silicon workload's code runs
natively on the Mac instead, and its pod proxies it:

    pod (a0 / fserve) -> bridge (host, 127.0.0.1) -> the same entrypoint, natively on the Mac -> Apple GPU

Every workload is its own host process, so several can run at once; they share the Mac's one GPU and its memory.
A workload that sets `FLYTE_APPLE_SILICON_EXECUTION=container` stays in its pod and gets a per-workload
`mlx_lm.server` on the host behind an OpenAI-compatible endpoint instead.

- `_bridge`: the host process `flyte start devbox --apple-silicon` runs. It authenticates each workload, runs and
  supervises its host process (or daemon), serves its status and logs, and tears it down.
- `_hostexec`: a workload's entrypoint on the host: the cached uv environment, the endpoint rewrites, the process.
- `_daemon`: one supervised `mlx_lm.server` (container mode) and its lifecycle states.
- `_client`: the pod side. The runtime hands the entrypoint to the bridge and mirrors it (logs, exit code, aborts,
  an app's port), or in container mode holds a lease on the workload's daemon.
- `_host`: what the `flyte` CLI on the Mac needs to start, stop and inspect the bridge.
- `_token`: the per-workload credential, minted by the backend (Go) and checked by the bridge.
"""
