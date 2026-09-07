"""
Run a plain script across multiple nodes via `flyte run python-script --clustered`.

`--clustered` wraps the script in a `flyte.clustered.ClusteredTaskEnvironment`
(a Kubernetes JobSet) instead of a single-pod `TaskEnvironment`: `--replicas`
pods each run `--nproc-per-node` processes, bootstrapped by `torchrun`. That
means this plain script sees the standard `torch.distributed` rendezvous env
vars (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT) already
populated — no `flyte` imports needed inside the script itself.

Run 2 nodes x 2 processes = world_size 4 (CPU-only via the "gloo" backend, no
GPUs required for this quick test):

    flyte run python-script examples/run_python_script/clustered_torchrun.py \\
        --packages torch --cpu 2 --memory 2Gi \\
        --clustered --replicas 2 --nproc-per-node 2

Follow to completion:

    flyte run --follow python-script examples/run_python_script/clustered_torchrun.py \\
        --packages torch --cpu 2 --memory 2Gi --clustered --replicas 2 --nproc-per-node 2

Add restart/failure-policy knobs, e.g. to survive a node eviction with one
free JobSet restart:

    flyte run python-script examples/run_python_script/clustered_torchrun.py \\
        --packages torch --clustered --replicas 2 --nproc-per-node 2 \\
        --cluster-max-restarts 1 --restart-on-host-maintenance

For a real multi-GPU DDP *training* example (not just a rendezvous smoke
test), see `examples/clustered/ddp_train.py`.
"""

import os

import torch
import torch.distributed as dist

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")  # gloo works CPU-only; use "nccl" when running with --gpu
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(f"[rank {rank}/{world_size}] local_rank={local_rank} master_addr={os.environ.get('MASTER_ADDR')}")

    # A trivial all-reduce proves the ranks can actually talk to each other
    # across pods, not just that torchrun set the env vars.
    tensor = torch.tensor([float(rank)])
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))
    print(f"[rank {rank}] all-reduce result={tensor.item()} (expected {expected})")
    assert tensor.item() == expected, "all-reduce mismatch: cross-pod rendezvous is broken"

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("clustered smoke test passed: all ranks rendezvoused and all-reduced correctly")
