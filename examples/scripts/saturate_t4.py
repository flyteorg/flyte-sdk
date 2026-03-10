"""
Saturate a T4 GPU for a few seconds with matrix multiplications, then verify results

You can run this using:

```bash
flyte run --follow python-script saturate_t4.py --gpu 1 --gpu-type T4 --packages torch
```
."""

import time

import torch


def main():
    assert torch.cuda.is_available(), "CUDA is not available"

    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    matrix_size = 4096
    duration_seconds = 5.0

    print(f"GPU: {name}")
    print(f"Saturating for ~{duration_seconds}s with {matrix_size}x{matrix_size} FP16 matmuls...")

    a = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    b = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)

    # Warm up
    torch.cuda.synchronize()
    _ = torch.mm(a, b)
    torch.cuda.synchronize()

    start = time.time()
    iterations = 0
    while time.time() - start < duration_seconds:
        _ = torch.mm(a, b)
        iterations += 1

    torch.cuda.synchronize()
    elapsed = time.time() - start

    flops_per_iter = 2 * matrix_size**3
    total_tflops = (iterations * flops_per_iter) / (elapsed * 1e12)

    print(f"Completed {iterations} matmuls in {elapsed:.2f}s")
    print(f"Throughput: ~{total_tflops:.1f} TFLOPS (FP16)")

    # Assertions
    assert iterations > 100, f"Expected >100 iterations, got {iterations}"
    assert elapsed >= 2.5, f"Expected ~5s runtime, got {elapsed:.2f}s"
    assert total_tflops > 5.0, f"Expected >5 TFLOPS, got {total_tflops:.1f}"
    print("All assertions passed!")


if __name__ == "__main__":
    main()
