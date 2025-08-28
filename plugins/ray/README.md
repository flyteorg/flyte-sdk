# Flyte Ray Plugin

Union can execute Ray jobs natively on a Kubernetes Cluster,
which manages a virtual cluster’s lifecycle, spin-up, and tear down.
It leverages the open-sourced KubeRay and can be enabled without signing up for any service.
This is like running a transient ray cluster —
a type of cluster spun up for a specific Ray job and torn down after completion.

To install the plugin, run the following command:

```bash
pip install --pre flyteplugins-ray
```
