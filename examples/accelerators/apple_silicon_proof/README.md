# Runbook: prove `gpu="AppleSilicon"` runs on the Apple Silicon GPU

This runbook shows that a devbox task or app with `flyte.Resources(gpu="AppleSilicon")` runs its code on the Mac's
Apple Silicon GPU, not on the CPU.

How it works: a devbox container is a Linux VM, and Linux has no Metal, so code inside the container could
never use the GPU. A `gpu="AppleSilicon"` workload therefore runs **natively on the Mac**. The devbox hands the
workload's entrypoint to a host-side bridge, which runs it in an environment built from the image's pip
packages. Because the code runs on the Mac, `proof.py` can gather the evidence itself, and you can confirm
it with tools that know nothing about Flyte.

## What it checks

`proof.py` fails unless every check passes:

| # | Check | Evidence |
|---|---|---|
| 1 | It runs natively on the Mac, not in the devbox container | `platform` reports `Darwin arm64` (the container is Linux) |
| 2 | MLX runs on an Apple Silicon Metal GPU | Metal device name, GPU architecture and core count |
| 3 | PyTorch's `mps` device works and beats the CPU | a 4096×4096 fp32 matmul on `mps` vs the CPU |
| 4 | MLX on the GPU is far faster than on the CPU | the same matmul on MLX's GPU vs CPU device |
| 5 | macOS reports the GPU busy during the GPU work | `Device Utilization %` from the IORegistry, sampled every 100 ms |

Check 5 reads the same counter Activity Monitor's GPU History draws.

## Prerequisites

- An Apple Silicon Mac (M1 or later) with Docker Desktop running.
- [`uv`](https://docs.astral.sh/uv/) on the Mac. The bridge uses it to build each workload's environment.
- A devbox image with the Apple Silicon backend. Until it ships in the published `flyte-devbox` image, build it from
  the flyte repo (`make -C docker/devbox-bundled build`) and pass `--image flyte-devbox:latest` to `flyte start devbox`.
  With an older image the bridge starts but the backend never uses it, and the task's pod stays `Pending`.

## Run it

1. Start a devbox that lets Apple Silicon workloads run on the Mac:

   ```bash
   flyte start devbox --apple-silicon
   ```

   This runs workload code outside any sandbox, so the flag also binds every devbox port to `127.0.0.1`: nothing
   else on your network can reach the devbox. It doesn't stop software already running on your Mac (including a web
   page making requests to `localhost`) from talking to the devbox's API, so don't browse untrusted sites while an
   Apple Silicon devbox is up. The ready panel ends with:

   ```
   🍎 Apple Silicon bridge on 127.0.0.1:30090: request it with flyte.Resources(gpu="AppleSilicon"); these workloads run natively on this Mac, devbox ports bound to 127.0.0.1
   ```

2. Point the CLI at the devbox: `flyte create config --devbox`.

3. Run the proof as a **task**:

   ```bash
   flyte run examples/accelerators/apple_silicon_proof/proof.py prove
   ```

   The first run builds the Mac's environment (torch, mlx) and takes a few minutes; later runs reuse it. The
   task's logs and its report tab show the verdict. Sample output from an M5 Max:

   ```
   PASS  Runs natively on the Mac, not in the devbox container: Darwin arm64 (the devbox container is a Linux VM)
   PASS  MLX runs on an Apple Silicon Metal GPU: Apple M5 Max (applegpu_g17s), 40 GPU cores
   PASS  PyTorch's mps device works and beats the CPU: 13.39 TFLOPS on mps vs 1.61 on the CPU (4096x4096 fp32 matmul)
   PASS  MLX on the GPU is far faster than on the CPU: 37.6 TFLOPS on the GPU vs 1.68 on the CPU
   PASS  macOS reports the GPU busy during the GPU work: mean GPU utilization 79.8% (MLX) and 92.7% (mps) vs 32.5% idle and 12.2% during the CPU matmul
   ```

4. Run the same proof as an **app** (`AppEnvironment`):

   ```bash
   flyte deploy examples/accelerators/apple_silicon_proof/proof.py app_env
   curl -s http://apple-silicon-proof-app-flytesnacks-development.localhost:30081/proof
   ```

   `/proof` answers `200` with `"verdict": "PASS"` and the same five checks. It answers `500` if any check fails.

Absolute numbers vary by chip. What matters is the gap: GPU throughput many times the CPU's, and GPU
utilization well above the idle baseline while the work runs.

## Confirm it yourself, without trusting this code

1. **Watch the GPU in Activity Monitor.** Open Activity Monitor → Window → GPU History (⌘4) and run the task
   again. The graph spikes while it runs.
2. **Read the kernel's counter in a terminal** while the task runs:

   ```bash
   while true; do ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'; sleep 0.5; done
   ```

3. **Find the process on the Mac.** `flyte get devbox` lists every Apple Silicon workload, shown as `runs on this host`.
   While the task runs, `ps -ef | grep 'apple-silicon/envs'` shows its Flyte entrypoint (`a0`) as a native macOS process.
   Its log is at `~/.flyte/apple-silicon/logs/<workload>.log` and also streams into the task's own logs.
4. **Negative control.** Recreate the devbox without `--apple-silicon` and run the task again:

   ```bash
   flyte delete devbox && flyte start devbox
   flyte run examples/accelerators/apple_silicon_proof/proof.py prove
   ```

   The task fails before it starts, with `APPLE_SILICON_NOT_AVAILABLE`. It does not quietly run on the CPU.

## Troubleshooting

| Symptom | Meaning |
|---|---|
| `APPLE_SILICON_NOT_AVAILABLE` | Either the devbox was started without `--apple-silicon` (`flyte start devbox --apple-silicon`), or the host isn't an Apple Silicon Mac. |
| `APPLE_SILICON_HOST_EXEC_FAILED: could not build the host environment` | `uv` is missing on the Mac, or a pip package has no macOS wheel. The task log has the `uv` output. |
| `APPLE_SILICON_BRIDGE_UNREACHABLE` | The container could not reach the bridge, or the bridge lost the workload's host process. Run `flyte get devbox`; if the bridge isn't running, run `flyte start devbox` and answer **N** to keep the devbox: its bridge restarts. |
| `APPLE_SILICON_DAEMON_OOM` | The workload's process on the Mac was killed for running out of memory. Several Apple Silicon workloads at once each load their own model: run fewer, or smaller. |
| The pod stays `Pending` | The devbox image predates the Apple Silicon backend; see Prerequisites. |
| Check 5 fails with a high idle baseline | Something else is using the GPU (a game, a video call, another model). Close it and rerun. |
