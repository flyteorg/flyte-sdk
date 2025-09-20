# Union PyTorch Plugin

Union can execute **PyTorch distributed training jobs** natively on a Kubernetes Cluster, which manages the lifecycle of worker pods, rendezvous coordination, spin-up, and tear down. It leverages the open-sourced **TorchElastic (torch.distributed.elastic)** launcher and the **Kubeflow PyTorch Operator**, enabling fault-tolerant and elastic training across multiple nodes.

This is like running a transient PyTorch cluster — worker groups are created for the specific job and torn down automatically after completion. Elastic training allows nodes to scale in and out, and failed workers can be restarted without bringing down the entire job.

To install the plugin, run the following command:

```bash
pip install --pre flyteplugins-pytorch
```

