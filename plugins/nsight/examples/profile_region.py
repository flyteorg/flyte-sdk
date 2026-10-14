"""Profile only a hot region of a GPU task, not the whole thing.

`@nsys_profile(capture="manual")` launches the task under Nsight Systems but collects nothing until
an `async with nsys.range(...)` block. Everything outside the range — model build, data prep, warmup
— runs unprofiled, so the trace and the report cover just the region you care about. This keeps a
long run's trace from ballooning to multiple gigabytes and drops the first-iteration lazy-loading
noise without any post-processing.

Each region collects independently: its own `.nsys-rep` download and its own section in the report
deck. Here the task profiles a "training" region and then a separate "evaluation" region, so you get
two clean traces from one task. Outside a profiling run (local execution) the ranges are transparent
no-ops, so the same code runs anywhere.

This runs remotely against local, unreleased code, so the image bakes the flyte and
flyteplugins-nsight wheels from ./dist. Build the wheels once, then run:

    make dist && FLYTE_PLUGIN_DIST=plugins/nsight make dist-plugins
    flyte run plugins/nsight/examples/profile_region.py profile_region
"""

from __future__ import annotations

import flyte

from flyteplugins.nsight import nsys, nsys_profile, nvtx

# Same base-image setup as profile_training.py (NGC PyTorch for nsys + torch, local wheels baked in,
# system-site-packages flipped so the venv sees NGC's torch). See that file for the full rationale.
image = (
    flyte.Image.from_base("nvcr.io/nvidia/pytorch:24.08-py3")
    .clone(extendable=True, name="nsight-region", python_version=(3, 10))
    .with_pip_packages("uv", "kubernetes")
    .with_local_v2()
    .with_local_v2_plugins(["flyteplugins-nsight"])
    .with_commands(
        ["sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' /opt/venv/pyvenv.cfg"]
    )
)

env = flyte.TaskEnvironment(
    name="nsight-region",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1"),
    # CAP_SYS_ADMIN + unconfined AppArmor — required for osrt tracing / GPU counters.
    pod_template=flyte.PodTemplate().allow_nested_sandboxing(),
)


@nsys_profile(capture="manual", trace=["cuda", "nvtx", "osrt"])
@env.task
async def profile_region(steps: int = 30, width: int = 4096, batch: int = 512) -> str:
    """Build and warm up a model unprofiled, then profile a training region and an eval region.

    capture="manual" means collection only happens inside the `async with nsys.range(...)` blocks;
    the setup and warmup above them are excluded from every trace.
    """
    import torch
    import torch.nn as nn

    dev = torch.device("cuda")
    model = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Build + warm up outside any range: none of this lands in a trace, which is the whole point of
    # manual capture (the alternative to the warmup-then-profile trick in profile_training.py).
    for _ in range(5):
        out = model(torch.randn(batch, width, device=dev))
        loss_fn(out, torch.randn(batch, width, device=dev)).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Region 1: the steady-state training loop. Writes its own .nsys-rep + report section.
    loss = torch.tensor(0.0)
    async with nsys.range("training"):
        for step in range(steps):
            with nvtx.range(f"step_{step}"):
                x = torch.randn(batch, width, device=dev)
                y = torch.randn(batch, width, device=dev)
                with nvtx.range("forward"):
                    out = model(x)
                    loss = loss_fn(out, y)
                with nvtx.range("backward"):
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
        torch.cuda.synchronize()

    # Region 2: evaluation, collected independently. A second .nsys-rep + report section, so you can
    # compare forward-only inference against the training loop without them sharing a trace.
    model.eval()
    async with nsys.range("evaluation"):
        with torch.inference_mode():
            for _ in range(10):
                with nvtx.range("eval_step"):
                    model(torch.randn(batch, width, device=dev))
        torch.cuda.synchronize()

    return f"training done, final loss: {loss.item():.4f}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(profile_region)
    print(run.url)
