import flyte
import flyte.app

import pathlib


# TODO:
# - ✅ Add postgres database for user data persistence
# - Set up external runners here: https://docs.n8n.io/hosting/configuration/task-runners/#setting-up-external-mode
# - Support python nodes: https://docs.n8n.io/code/code-node/#python-native
# - Add support for Flyte nodes: https://docs.n8n.io/hosting/configuration/task-runners/#adding-extra-dependencies
n8n_app_image = (
    flyte.Image.from_base("node:24-slim")
    .clone(name="n8n-app-image")
    .with_pip_packages("flyte==2.0.0b54")
    .with_apt_packages("ca-certificates", "curl", "gnupg", "npm")
    .with_commands(["npm install -g n8n@2.4.8"])
)

launcher_url = "https://github.com/n8n-io/task-runner-launcher/releases/download/1.4.2/task-runner-launcher-1.4.2-linux-amd64.tar.gz"

n8n_task_runner_image = (
    flyte.Image.from_base("node:24-slim")
    .clone(name="n8n-task-runner-image")
    .with_pip_packages("flyte==2.0.0b54")
    .with_apt_packages("ca-certificates", "curl", "gnupg", "npm")
    .with_commands([
        f"curl -L -o /tmp/task-runner-launcher.tar.gz {launcher_url}",
        "tar -xzf /tmp/task-runner-launcher.tar.gz -C /usr/local/bin",
        "chmod +x /usr/local/bin/task-runner-launcher",
        "rm /tmp/task-runner-launcher.tar.gz",
    ])
    .with_source_file(pathlib.Path(__file__).parent / "n8n-task-runners.json", "/etc/n8n-task-runners.json")
)

n8n_app = flyte.app.AppEnvironment(
    name="n8n-app",
    image=n8n_app_image,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    port=5678,
    args=["n8n", "start"],
    secrets=[
        flyte.Secret("n8n_postgres_password", as_env_var="DB_POSTGRESDB_PASSWORD"),
        flyte.Secret("n8n_postgres_host", as_env_var="DB_POSTGRESDB_HOST"),
    ],
    env_vars={
        # db config
        "DB_POSTGRESDB_DATABASE": "postgres",
        "DB_POSTGRESDB_USER": "postgres",
        "DB_POSTGRESDB_PORT": "5432",

        # external runner config
        "N8N_RUNNERS_ENABLED": "true",
        "N8N_RUNNERS_TYPE": "external",
        "N8N_RUNNERS_BROKER_LISTEN_ADDRESS": "0.0.0.0",
        "N8N_RUNNERS_AUTH_TOKEN": "test-token",
        "N8N_NATIVE_PYTHON_RUNNER": "true",
    }
)

n8n_task_runner = flyte.app.AppEnvironment(
    name="n8n-task-runner",
    image=n8n_task_runner_image,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    port=5678,
    args=["/usr/local/bin/task-runner-launcher", "javascript", "python"],
    env_vars={
        "N8N_RUNNERS_AUTH_TOKEN": "test-token",
    }
)

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(n8n_task_runner)
    print(app.url)
