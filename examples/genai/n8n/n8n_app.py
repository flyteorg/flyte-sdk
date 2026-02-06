import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment
import pathlib

import kubernetes

from fastapi import FastAPI


N8N_VERSION = "2.4.8"
N8N_RUNNERS_AUTH_TOKEN = "your-secret-here"

n8n_with_runner_pod_template = flyte.PodTemplate(
    primary_container_name="app",
    pod_spec=kubernetes.client.V1PodSpec(
        containers=[
            # Primary container: n8n main server
            kubernetes.client.V1Container(
                name="app",
                image=f"n8nio/n8n:{N8N_VERSION}",
                ports=[
                    kubernetes.client.V1ContainerPort(container_port=5678),
                ],
                env=[
                    kubernetes.client.V1EnvVar(name="N8N_RUNNERS_ENABLED", value="true"),
                    kubernetes.client.V1EnvVar(name="N8N_RUNNERS_MODE", value="external"),
                    kubernetes.client.V1EnvVar(name="N8N_RUNNERS_BROKER_LISTEN_ADDRESS", value="0.0.0.0"),
                    kubernetes.client.V1EnvVar(name="N8N_RUNNERS_AUTH_TOKEN", value=N8N_RUNNERS_AUTH_TOKEN),
                    kubernetes.client.V1EnvVar(name="N8N_NATIVE_PYTHON_RUNNER", value="true"),
                ],
                volume_mounts=[
                    kubernetes.client.V1VolumeMount(
                        name="n8n-data",
                        mount_path="/home/node/.n8n",
                    ),
                ],
            ),
            # Sidecar container: task runners
            kubernetes.client.V1Container(
                name="task-runners",
                # image=f"n8nio/runners:{N8N_VERSION}",
                image="ghcr.io/flyteorg/n8n-task-runner-image:896c4478822858a314074b1a3caf882a",
                env=[
                    # Connect to n8n broker via localhost since they're in the same pod
                    kubernetes.client.V1EnvVar(name="N8N_RUNNERS_TASK_BROKER_URI", value="http://localhost:5679"),
                    kubernetes.client.V1EnvVar(name="N8N_RUNNERS_AUTH_TOKEN", value=N8N_RUNNERS_AUTH_TOKEN),
                ],
            ),
        ],
        volumes=[
            kubernetes.client.V1Volume(
                name="n8n-data",
                empty_dir=kubernetes.client.V1EmptyDirVolumeSource(),
            ),
        ],
    )
)


# TODO:
# - ✅ Add postgres database for user data persistence
# - Set up external runners here: https://docs.n8n.io/hosting/configuration/task-runners/#setting-up-external-mode
# - Support python nodes: https://docs.n8n.io/code/code-node/#python-native
# - Add support for Flyte nodes: https://docs.n8n.io/hosting/configuration/task-runners/#adding-extra-dependencies
n8n_app_image = (
    flyte.Image.from_base("node:24-slim")
    .clone(name="n8n-app-image")
    .with_pip_packages("flyte==2.0.0b54", "fastapi", "uvicorn")
    .with_apt_packages("ca-certificates", "curl", "gnupg", "npm")
    .with_commands(["npm install -g n8n@2.4.8"])
)

# launcher_url = "https://github.com/n8n-io/task-runner-launcher/releases/download/1.4.2/task-runner-launcher-1.4.2-linux-amd64.tar.gz"

# n8n_task_runner_image = (
#     flyte.Image.from_base("node:24-slim")
#     .clone(name="n8n-task-runner-image")
#     .with_pip_packages("flyte==2.0.0b54", "kubernetes")
#     .with_apt_packages("ca-certificates", "curl", "gnupg", "npm")
#     # install the task-runner-launcher: https://github.com/n8n-io/task-runner-launcher
#     .with_commands([
#         f"curl -L -o /tmp/task-runner-launcher.tar.gz {launcher_url}",
#         "tar -xzf /tmp/task-runner-launcher.tar.gz -C /usr/local/bin",
#         "chmod +x /usr/local/bin/task-runner-launcher",
#         "rm /tmp/task-runner-launcher.tar.gz",
#     ])
#     .with_source_file(pathlib.Path(__file__).parent / "n8n-task-runners.json", "/etc/n8n-task-runners.json")
# )

bump = "4"
n8n_app = flyte.app.AppEnvironment(
    name="n8n-app",
    image=n8n_app_image,
    pod_template=n8n_with_runner_pod_template,
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    port=5678,
    command=["n8n", "start"],
    secrets=[
        flyte.Secret("n8n_postgres_password", as_env_var="DB_POSTGRESDB_PASSWORD"),
        flyte.Secret("n8n_postgres_host", as_env_var="DB_POSTGRESDB_HOST"),
    ],
    requires_auth=False,
    env_vars={
        "N8N_ENCRYPTION_KEY": "abc123",
        
        # db config
        "DB_POSTGRESDB_DATABASE": "postgres",
        "DB_POSTGRESDB_USER": "postgres",
        "DB_POSTGRESDB_PORT": "5432",

        "BUMP": bump,
    }
)


app = FastAPI()

@app.get("/")
async def root():
    import os

    return {"message": "Hello World", "n8n_app_endpoint": os.getenv("N8N_APP_URL")}

n8n_debugger = FastAPIAppEnvironment(
    app=app,
    name="n8n-debugger",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "kubernetes"),
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    port=8080,
    requires_auth=False,
    depends_on=[n8n_app],
    parameters=[
        flyte.app.Parameter(
            name="n8n_app_endpoint",
            value=flyte.app.AppEndpoint(app_name="n8n-app", public=False),
            env_var="N8N_APP_URL",
        ),
    ],
)

# n8n_task_runner = flyte.app.AppEnvironment(
#     name="n8n-task-runner",
#     image="ghcr.io/flyteorg/n8n-task-runner-image:896c4478822858a314074b1a3caf882a",
#     resources=flyte.Resources(cpu=2, memory="2Gi"),
#     port=5678,
#     command=["/usr/local/bin/task-runner-launcher", "javascript", "python"],
#     requires_auth=False,
#     env_vars={
#         "N8N_RUNNERS_LAUNCHER_LOG_LEVEL": "debug",
#         "N8N_RUNNERS_TASK_BROKER_URI": "http://n8n-app.flytesnacks-development.svc.cluster.local:5679",
#         "N8N_RUNNERS_AUTH_TOKEN": "test-token",

#         "BUMP": bump,
#     },
#     depends_on=[n8n_app, n8n_debugger],
# )

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(n8n_debugger)
    print(app.url)
