import time

import torch
from torch import Tensor

import flyte
import flyte.git


gpu_env = flyte.TaskEnvironment(
    name="gpu_env", 
    image=flyte.Image.from_debian_base(name="gpu_image").with_pip_packages("torch"),
    resources=flyte.Resources(gpu=1, memory="2Gi")
    )


@gpu_env.task
def timed_gpu_matmul(m: int, k:int, n:int, duration_sec: int = 60) -> list[list[float]]:

    a = torch.randn(m, k)
    b = torch.randn(k, n)

    """Repeat (a @ b) on GPU for ~duration_sec seconds using the new torch.amp.autocast API."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")
    
    # Move to GPU
    A = torch.as_tensor(a, device="cuda", dtype=torch.float16)
    B = torch.as_tensor(b, device="cuda", dtype=torch.float16)
    
    start = time.time()
    last_C = None
    while time.time() - start < duration_sec:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            C = A @ B
        # Ensure the GPU work is done before checking time
        torch.cuda.synchronize()
        last_C = C
    
    # Move to CPU, convert to float32, then to nested list
    result_tensor: Tensor = last_C.to(torch.float32).cpu()
    result_list: list[list[float]] = result_tensor.tolist()

    raise Exception("test debug button")

    return result_list


if __name__ == "__main__":
    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(timed_gpu_matmul, m=1024, k=2048, n=512, duration_sec=120)
    print(run.url)