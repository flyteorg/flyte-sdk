"""
Devbox-local Apple Silicon accelerator, requested with `flyte.Resources(gpu="AppleSilicon")`.

A devbox container is a Linux VM with no access to the Mac's Metal GPU, and MLX cannot run inside it.
So the ML work runs in a host-side MLX daemon (`mlx_lm.server`) and the container reaches it over a local,
OpenAI-compatible endpoint:

    container -> bridge (host, 127.0.0.1) -> per-workload mlx_lm.server (Metal) -> Apple GPU

- `_bridge`: the host process `flyte start devbox` runs. It authenticates each workload, starts, supervises
  and tears down one daemon per workload, and proxies the workload's OpenAI requests to it.
- `_daemon`: one supervised `mlx_lm.server` and its lifecycle states.
- `_client`: the container side. The Flyte runtime attaches to the bridge before user code runs, holds a
  lease on the daemon for the workload's lifetime, and streams the daemon's logs into the container's.
- `_host`: what the `flyte` CLI on the Mac needs to start, stop and inspect the bridge.
- `_token`: the per-workload credential, minted by the backend (Go) and checked by the bridge.
"""
