# Union Dask Plugin

Flyte can execute `dask` jobs natively on a Kubernetes Cluster, which manages the virtual `dask` cluster's lifecycle
(spin-up and tear down). It leverages the open-source Kubernetes Dask Operator and can be enabled without signing up
for any service. This is like running a transient (ephemeral) `dask` cluster - a type of cluster spun up for a specific
task and torn down after completion. This helps in making sure that the Python environment is the same on the job-runner
(driver), scheduler and the workers.

To install the plugin, run the following command:

```bash
pip install --pre flyteplugins-dask
```
