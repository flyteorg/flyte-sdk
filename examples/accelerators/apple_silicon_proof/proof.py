"""
Prove that a devbox task or app with gpu="AppleSilicon" runs its code on the Mac's Apple Silicon GPU.

See README.md in this directory for the runbook. In short:

    flyte start devbox --apple-silicon
    flyte run examples/accelerators/apple_silicon_proof/proof.py prove        # a task
    flyte deploy examples/accelerators/apple_silicon_proof/proof.py app_env   # an app: GET /proof

A gpu="AppleSilicon" workload runs natively on the Mac (a devbox container is a Linux VM with no Metal), so this code
gathers the evidence itself and fails unless all of it checks out:

1. It is running on macOS on Apple Silicon, not in the devbox's Linux container.
2. MLX runs on an Apple Silicon Metal GPU.
3. PyTorch's mps device works, and is far faster than the CPU.
4. The same MLX matmul is far faster on the GPU than on the CPU.
5. macOS itself reports the GPU busy while that work runs (the counter Activity Monitor's GPU History draws).
"""

import json
import platform
import re
import subprocess
import threading
import time

import flyte
import flyte.report
from flyte.app import AppEnvironment

image = flyte.Image.from_debian_base().with_pip_packages("torch", "mlx; sys_platform == 'darwin'")
resources = flyte.Resources(cpu=1, memory="2Gi", gpu="AppleSilicon")

env = flyte.TaskEnvironment(name="apple-silicon-proof", image=image, resources=resources)

# The same proof for a long-running app: GET /proof runs every check from inside the app.
app_env = AppEnvironment(
    name="apple-silicon-proof-app", image=image, resources=resources, port=8080, requires_auth=False
)


def _ioreg() -> str:
    try:
        return subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return ""


def gpu_utilization() -> int | None:
    """The GPU utilization (%) macOS reports in the IORegistry, or None where there is none (e.g. Linux)."""
    m = re.search(r'"Device Utilization %"=(\d+)', _ioreg())
    return int(m.group(1)) if m else None


def gpu_cores() -> int | None:
    m = re.search(r'"gpu-core-count" = (\d+)', _ioreg())
    return int(m.group(1)) if m else None


class Sampler:
    """Samples macOS's GPU utilization every 100 ms, labelling each sample with the current phase."""

    def __init__(self):
        self.phase = "idle"
        self.samples: dict[str, list[int]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            if (value := gpu_utilization()) is not None:
                self.samples.setdefault(self.phase, []).append(value)
            self._stop.wait(0.1)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def mean(self, phase: str) -> float:
        values = self.samples.get(phase) or [0]
        return round(sum(values) / len(values), 1)


def tflops(run, n: int, seconds: float) -> float:
    run()  # warm up
    runs, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        run()
        runs += 1
    return round(2 * n**3 * runs / (time.perf_counter() - start) / 1e12, 2)


def gather(n: int = 4096, seconds: float = 2.0) -> dict:
    import mlx.core as mx
    import torch

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    a, b = mx.random.normal((n, n)), mx.random.normal((n, n))
    ta, tb = torch.randn(n, n, device="mps"), torch.randn(n, n, device="mps")
    ca, cb = ta.cpu(), tb.cpu()
    with Sampler() as sampler:
        time.sleep(1.0)  # an idle baseline
        sampler.phase = "mlx_gpu"
        mlx_gpu = tflops(lambda: mx.eval(mx.matmul(a, b, stream=mx.gpu)), n, seconds)
        sampler.phase = "mlx_cpu"
        mlx_cpu = tflops(lambda: mx.eval(mx.matmul(a, b, stream=mx.cpu)), n, seconds)
        sampler.phase = "torch_mps"
        torch_mps = tflops(lambda: (ta @ tb).sum().item(), n, seconds)
        sampler.phase = "torch_cpu"
        torch_cpu = tflops(lambda: (ca @ cb).sum().item(), n, seconds)
        sampler.phase = "after"
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "metal": {
            "available": mx.metal.is_available(),
            "device": info.get("device_name"),
            "architecture": info.get("architecture"),
            "gpu_cores": gpu_cores(),
        },
        "torch_mps_available": torch.backends.mps.is_available(),
        "tflops": {"mlx_gpu": mlx_gpu, "mlx_cpu": mlx_cpu, "torch_mps": torch_mps, "torch_cpu": torch_cpu},
        "gpu_utilization_mean": {p: sampler.mean(p) for p in ("idle", "mlx_gpu", "mlx_cpu", "torch_mps")},
    }


def verdict(e: dict) -> list[tuple[str, bool, str]]:
    t, u, metal = e["tflops"], e["gpu_utilization_mean"], e["metal"]
    return [
        (
            "Runs natively on the Mac, not in the devbox container",
            e["platform"] == "Darwin arm64",
            f"{e['platform']} (the devbox container is a Linux VM)",
        ),
        (
            "MLX runs on an Apple Silicon Metal GPU",
            bool(metal["available"]) and str(metal["device"]).startswith("Apple"),
            f"{metal['device']} ({metal['architecture']}), {metal['gpu_cores']} GPU cores",
        ),
        (
            "PyTorch's mps device works and beats the CPU",
            e["torch_mps_available"] and t["torch_mps"] >= 3 * t["torch_cpu"],
            f"{t['torch_mps']} TFLOPS on mps vs {t['torch_cpu']} on the CPU (4096x4096 fp32 matmul)",
        ),
        (
            "MLX on the GPU is far faster than on the CPU",
            t["mlx_gpu"] >= 5 * t["mlx_cpu"],
            f"{t['mlx_gpu']} TFLOPS on the GPU vs {t['mlx_cpu']} on the CPU",
        ),
        (
            "macOS reports the GPU busy during the GPU work",
            min(u["mlx_gpu"], u["torch_mps"]) >= u["idle"] + 25,
            f"mean GPU utilization {u['mlx_gpu']}% (MLX) and {u['torch_mps']}% (mps) vs {u['idle']}% idle "
            f"and {u['mlx_cpu']}% during the CPU matmul",
        ),
    ]


CARD = "background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 20px;margin:0 0 16px"
MUTED = "color:#6b7280;font-size:13px"


def _bars(title: str, unit: str, values: list[tuple[str, float, str]]) -> str:
    """A small horizontal bar chart: (label, value, color) rows scaled to the largest value."""
    top = max(v for _, v, _ in values) or 1
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0">'
        f'<div style="width:110px;{MUTED}">{label}</div>'
        f'<div style="flex:1;background:#f3f4f6;border-radius:6px;height:18px">'
        f'<div style="width:{max(v / top * 100, 1):.1f}%;background:{color};height:18px;border-radius:6px"></div></div>'
        f'<div style="width:110px;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums">'
        f"{v:g} {unit}</div></div>"
        for label, v, color in values
    )
    return f'<div style="{CARD}"><div style="font-weight:600;margin-bottom:8px">{title}</div>{rows}</div>'


def render(e: dict, checks: list[tuple[str, bool, str]]) -> str:
    passed = all(ok for _, ok, _ in checks)
    t, u, metal = e["tflops"], e["gpu_utilization_mean"], e["metal"]
    badge = (
        f'<span style="background:{"#dcfce7" if passed else "#fee2e2"};color:{"#166534" if passed else "#991b1b"};'
        f'border-radius:999px;padding:4px 12px;font-weight:600;font-size:14px">'
        f"{'PASS' if passed else 'FAIL'} · {sum(ok for _, ok, _ in checks)}/{len(checks)} checks</span>"
    )
    header = (
        f'<div style="{CARD};display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">'
        f'<div><div style="font-size:20px;font-weight:700">Apple Silicon proof</div>'
        f'<div style="{MUTED}">{metal["device"]} · {metal["architecture"]} · {metal["gpu_cores"]} GPU cores · '
        f"{e['platform']}</div></div>{badge}</div>"
    )
    rows = "".join(
        f'<tr style="border-top:1px solid #f3f4f6"><td style="padding:8px 8px 8px 0;font-size:16px">'
        f"{'✅' if ok else '❌'}</td>"
        f'<td style="padding:8px;font-weight:600">{name}</td><td style="padding:8px;{MUTED}">{detail}</td></tr>'
        for name, ok, detail in checks
    )
    table = f'<div style="{CARD}"><table style="width:100%;border-collapse:collapse">{rows}</table></div>'
    charts = (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:0 16px">'
        + _bars(
            "Matmul throughput (4096x4096 fp32)",
            "TFLOPS",
            [
                ("MLX · GPU", t["mlx_gpu"], "#7652a2"),
                ("MLX · CPU", t["mlx_cpu"], "#c4b5fd"),
                ("torch · mps", t["torch_mps"], "#2563eb"),
                ("torch · CPU", t["torch_cpu"], "#93c5fd"),
            ],
        )
        + _bars(
            "GPU utilization reported by macOS",
            "%",
            [
                ("idle", u["idle"], "#d1d5db"),
                ("MLX · GPU", u["mlx_gpu"], "#7652a2"),
                ("MLX · CPU", u["mlx_cpu"], "#c4b5fd"),
                ("torch · mps", u["torch_mps"], "#2563eb"),
            ],
        )
        + "</div>"
    )
    evidence = (
        f'<details style="{CARD}"><summary style="cursor:pointer;font-weight:600">Raw evidence</summary>'
        f'<pre style="margin:12px 0 0;font-size:12px">{json.dumps(e, indent=2)}</pre></details>'
    )
    return (
        '<div style="font-family:system-ui,-apple-system,sans-serif;color:#111827;background:#f9fafb;'
        f'padding:20px;max-width:960px">{header}{table}{charts}{evidence}</div>'
    )


@env.task(report=True)
async def prove() -> dict:
    evidence = gather()
    checks = verdict(evidence)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    flyte.report.get_tab("main").log(render(evidence, checks))
    await flyte.report.flush.aio()
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        raise RuntimeError(f"Apple Silicon proof failed: {failed}")
    return {name: detail for name, _, detail in checks}


@app_env.server
def serve_proof():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/proof":
                status, payload = (200, {"status": "ok"}) if self.path == "/health" else (404, {"try": "/proof"})
            else:
                evidence = gather()
                checks = verdict(evidence)
                status = 200 if all(ok for _, ok, _ in checks) else 500
                payload = {
                    "verdict": "PASS" if status == 200 else "FAIL",
                    "checks": [{"check": name, "pass": ok, "detail": detail} for name, ok, detail in checks],
                    "evidence": evidence,
                }
            body = json.dumps(payload, indent=2).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    # 127.0.0.1: on the host, only the devbox's port forward needs to reach the app.
    ThreadingHTTPServer(("127.0.0.1", app_env.get_port().port), Handler).serve_forever()


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(prove)
    print(run.url)
    run.wait()
    print(run.outputs())
