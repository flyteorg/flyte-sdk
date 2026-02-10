import flyte
import flyte.app
import pathlib

import kubernetes


def n8n_pod_template(version: str, runner_auth_token: str, runner_image_uri: str | None = None) -> flyte.PodTemplate:
    return flyte.PodTemplate(
        primary_container_name="app",
        pod_spec=kubernetes.client.V1PodSpec(
            containers=[
                # Primary container: n8n main server
                kubernetes.client.V1Container(name="app", image=f"n8nio/n8n:{version}"),
                # Sidecar container: task runners
                kubernetes.client.V1Container(
                    name="task-runners",
                    image=runner_image_uri or f"n8nio/runners:{version}",
                    env=[
                        # Connect to n8n broker via localhost since they're in the same pod
                        kubernetes.client.V1EnvVar(name="N8N_RUNNERS_TASK_BROKER_URI", value="http://127.0.0.1:5679"),
                        kubernetes.client.V1EnvVar(name="N8N_RUNNERS_AUTH_TOKEN", value=runner_auth_token),
                    ],
                ),
            ],
        )
    )


n8n_app = flyte.app.AppEnvironment(
    name="n8n-app",
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 1)),
    port=5678,
    command=["n8n", "start"],
    secrets=[
        flyte.Secret("n8n_postgres_password", as_env_var="DB_POSTGRESDB_PASSWORD"),
        flyte.Secret("n8n_encryption_key", as_env_var="N8N_ENCRYPTION_KEY"),
    ],
    requires_auth=False,
    env_vars={
        "N8N_RUNNERS_ENABLED": "true",
        "N8N_RUNNERS_MODE": "external",
        "N8N_RUNNERS_BROKER_LISTEN_ADDRESS": "0.0.0.0",
        "N8N_NATIVE_PYTHON_RUNNER": "true",

        # db config: https://docs.n8n.io/hosting/installation/docker/#using-with-postgresql
        "DB_TYPE": "postgresdb",
        "DB_POSTGRESDB_HOST": "aws-0-us-west-2.pooler.supabase.com",
        "DB_POSTGRESDB_DATABASE": "postgres",
        "DB_POSTGRESDB_USER": "postgres.qcfcidgymclxvslgphyb",
        "DB_POSTGRESDB_PORT": "6543",
        "DB_POSTGRESDB_SCHEMA": "public",
    }
)


def build_runner_image() -> flyte.Image:
    flyte.init_from_config(image_builder="local")

    image = flyte.Image.from_dockerfile(
        pathlib.Path(__file__).parent / "task_runner.dockerfile",
        registry="ghcr.io/flyteorg",
        name="n8n-task-runner-image",
    )
    return flyte.build(image, wait=True)


def get_webhook_url(subdomain: str) -> str:
    cfg = get_init_config()
    return f"https://{subdomain}.apps.{cfg.client.endpoint.replace('dns:///', '').rstrip('/')}/"


if __name__ == "__main__":
    import random
    import string

    from flyte._initialize import get_init_config

    n8n_version = "2.6.3"
    # Create a random 32-character alphanumeric string for the n8n runners auth token. it's okay
    # to regenerate this every time the app is deployed, since only the main n8n app and the runner
    # sidecar container use this token.
    n8n_runners_auth_token = ''.join(random.choices(string.ascii_letters + string.digits, k=32))

    image = build_runner_image()

    flyte.init_from_config(image_builder="remote")
    pod_template = n8n_pod_template(
        version=n8n_version,
        runner_auth_token=n8n_runners_auth_token,
        runner_image_uri=image.uri,
    )

    subdomain = "n8n-app"
    webhook_url = get_webhook_url(subdomain)
    
    app = flyte.serve(
        n8n_app.clone_with(
            name="n8n-app-with-runners",
            pod_template=pod_template,
            domain=flyte.app.Domain(subdomain=subdomain),
            env_vars=n8n_app.env_vars | {
                "WEBHOOK_URL": webhook_url,
                "N8N_RUNNERS_AUTH_TOKEN": n8n_runners_auth_token,
            },
        )
    )
    print(app.url)
