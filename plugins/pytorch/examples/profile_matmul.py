"""GPU profiling with torch.profiler, rendered in the Flyte report.

The task runs a few matmul+add steps under `torch_profile()`; the resulting report tab shows
summary tables and an interactive Perfetto timeline (ProfilerStep#N / matmul_add / aten::mm
spans on CPU and CUDA tracks) right in the Flyte UI.

Run:
    make dist && FLYTE_PLUGIN_DIST=plugins/pytorch make dist-plugins
    flyte run plugins/pytorch/examples/profile_matmul.py profile_matmul
"""

import flyte

from flyteplugins.pytorch import torch_profile

image = (
    flyte.Image.from_debian_base(name="torch-profile", python_version=(3, 12))
    .with_pip_packages("torch")
    .with_local_v2()
    .with_local_v2_plugins(["flyteplugins-pytorch"])
)

env = flyte.TaskEnvironment(
    name="torch-profile",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1"),
)


@env.task(report=True)
def profile_matmul(steps: int = 8, n: int = 4096) -> str:
    import torch
    from torch.profiler import record_function, schedule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # repeat=1: keep the completed profiling cycle — without it, stepping past the active
    # window starts a new cycle and clears the collected events (empty report).
    with torch_profile(profile_memory=True, schedule=schedule(wait=1, warmup=2, active=4, repeat=1)) as prof:
        for _ in range(steps):
            with record_function("matmul_add"):
                c = a @ b + a
            if device == "cuda":
                torch.cuda.synchronize()
            prof.step()

    return f"done on {device}: {c.shape}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(profile_matmul)
    print(run.url)
