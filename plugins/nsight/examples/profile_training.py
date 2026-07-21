"""Profile a GPU training task with Nsight Systems, natively.

Adding @nsys_profile is the whole integration: the task runs under nsys,
its metrics land in the Flyte report, and the .nsys-rep comes back as a downloadable output.

This runs remotely against local, unreleased code, so the image bakes the flyte and
flyteplugins-nsight wheels from ./dist rather than pulling them from PyPI. Build the wheels once,
then run:

    make dist && FLYTE_PLUGIN_DIST=plugins/nsight make dist-plugins
    flyte run plugins/nsight/examples/profile_training.py profile_training
"""

from __future__ import annotations

import flyte

from flyteplugins.nsight import nsys_profile, nvtx

# NGC PyTorch ships torch and the Nsight Systems CLI (nsys) preinstalled. Making an arbitrary base
# image flyte-runnable needs .clone(extendable=True, name=...) and a matching python_version. uv is
# for the entrypoint's `uv run`; kubernetes because allow_nested_sandboxing() imports
# kubernetes.client when this module is imported inside the task. See the accelerators example for
# the full base-image rationale.
image = (
    flyte.Image.from_base("nvcr.io/nvidia/pytorch:24.08-py3")
    .clone(extendable=True, name="nsight-train", python_version=(3, 10))
    .with_pip_packages("uv", "kubernetes")
    # Bake the locally-built flyte and flyteplugins-nsight wheels from ./dist into the image so the
    # remote run exercises unreleased core changes (src/flyte/_bin/runtime.py) and the unpublished
    # plugin, not PyPI releases. with_local_v2 must come first so flyte is already installed before
    # the plugin's dependencies resolve.
    .with_local_v2()
    .with_local_v2_plugins(["flyteplugins-nsight"])
    # NGC installs torch into the system Python, but flyte runs the task in an isolated /opt/venv,
    # so `import torch` fails there. Flip the venv to include system site-packages: torch resolves to
    # NGC's optimized build while flyte's own deps (installed into the venv) still take precedence.
    # Avoids a ~2GB torch reinstall. uv writes this exact line into pyvenv.cfg, so the sed is reliable.
    .with_commands(
        ["sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' /opt/venv/pyvenv.cfg"]
    )
)

env = flyte.TaskEnvironment(
    name="nsight-training",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1"),
    # CAP_SYS_ADMIN + unconfined AppArmor — required for osrt tracing / GPU counters.
    pod_template=flyte.PodTemplate().allow_nested_sandboxing(),
)


@nsys_profile(trace=["cuda", "nvtx", "osrt"])
@env.task
async def profile_training(steps: int = 20, width: int = 4096, batch: int = 512) -> str:
    """A tiny GPU training loop, annotated with NVTX ranges, profiled end to end.

    Swap the body for your real training code (Lightning, HF Trainer, etc.); keep a few nvtx.range
    markers so the timeline and the NVTX summary have labelled regions to show.
    """
    import torch
    import torch.nn as nn

    dev = torch.device("cuda")
    model = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Warm up before the profiled loop so the NVTX steps reflect steady state, not one-time lazy
    # loading (CUDA context, cuDNN/cutlass autotune, module load). Without this, step_0 alone runs
    # ~1000x a warm step and dominates the timeline and the report. These iterations are still under
    # nsys but carry no NVTX ranges, so they stay out of the step_/forward/backward summary.
    for _ in range(3):
        out = model(torch.randn(batch, width, device=dev))
        loss = loss_fn(out, torch.randn(batch, width, device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    loss = torch.tensor(0.0)
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
    return f"training done, final loss: {loss.item():.4f}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(profile_training)
    print(run.url)
