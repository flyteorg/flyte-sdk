import os

from kubernetes.client import V1Container, V1EmptyDirVolumeSource, V1EnvVar, V1PodSpec, V1Volume, V1VolumeMount

import flyte

# Create a custom pod template with AWS secrets sidecar
# This logic can be encapsulated into a reusable function within a library for broader use.
pod_template = flyte.PodTemplate(
  primary_container_name="primary",
  pod_spec=V1PodSpec(
    # Replace with any Kubernetes service account name that exists within the namespace.
    service_account_name="default",

    init_containers=[
        # Replace with your preferred sidecar image that can fetch secrets.
        V1Container(
            name="secrets-fetcher",
            image="amazon/aws-cli:latest",
            command=["/bin/sh", "-c"],
            args=[
                """
                    echo "Fetching secrets from AWS..."

                    # Fetch secret from AWS Secrets Manager
                    aws secretsmanager get-secret-value \
                      --secret-id mike-test/test-not-a-real-secret \
                      --region us-west-2 \
                      --query SecretString \
                      --output text > /secrets/secret.txt

                    echo "Secrets fetched successfully at $(date)"
                    ls -la /secrets/
                    """
            ],
            volume_mounts=[
                V1VolumeMount(
                    name="secrets-volume",
                    mount_path="/secrets"
                )
            ],
            env=[
                V1EnvVar(name="AWS_REGION", value="us-west-2")
            ]
        )
    ],

    containers=[
        # Primary Flyte task container
        V1Container(
            name="primary",
            env=[
                V1EnvVar(name="SECRET_FILE_PATH",
                         value="/etc/secrets/secret.txt"),
            ],

            # Mount the same volume to access the fetched secrets
            volume_mounts=[
                V1VolumeMount(
                    name="secrets-volume",
                    mount_path="/etc/secrets",
                    read_only=True  # Read-only for security
                )
            ]
        ),
    ],

    # Define the shared volume for sharing secrets between sidecar and primary container
    volumes=[
        V1Volume(
            name="secrets-volume",
            empty_dir=V1EmptyDirVolumeSource(
                medium="Memory"
            )
        )
    ]
  )
)

image = flyte.Image.from_debian_base(
    name="secret-sidecar-image-2",
).with_pip_packages("kubernetes")

env = flyte.TaskEnvironment(
  name="side_car_secret_fetcher",
  image=image,
  pod_template=pod_template,
  resources=flyte.Resources(memory="250Mi"),
)

@env.task
def fn(env_var: str) -> str:  # type annotations are recommended.
    file_path = os.getenv(env_var)
    with open(file_path, "r") as f:
        secret_content = f.read().strip()
    return f"File from env {env_var} has secret contents: {secret_content}"


# tasks can also call other tasks, which will be manifested in different containers.
@env.task
def main(env_vars: list[str]) -> list[str]:
    return list(flyte.map(fn, env_vars))


if __name__ == "__main__":
    flyte.init_from_config()  # establish remote connection from within your script.
    run = flyte.run(main, env_vars=["SECRET_FILE_PATH"])  # run remotely inline and pass data.

    # print various attributes of the run.
    print(run.name)
    print(run.url)

    run.wait()  # stream the logs from the root action to the terminal.
