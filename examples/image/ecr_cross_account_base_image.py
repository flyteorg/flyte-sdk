import flyte
from flyte import Image

"""
This example builds an image on top of a private base image that lives in an Amazon ECR registry in a
*different* AWS account from your data plane.

Without a secret, the remote image builder only authenticates to its own default registry, so the pull of the
base image is anonymous and fails with `401 Unauthorized`. IAM policies alone can't fix that, because no
credentials are ever sent.

1. Allow the data plane's AWS account to pull from the repository by adding an ECR repository policy in the
   account that owns the image (replace <DATA_PLANE_ACCOUNT_ID>):
```
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "AllowDataPlanePull",
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::<DATA_PLANE_ACCOUNT_ID>:root"},
    "Action": ["ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]
  }]
}
```

2. Save a Docker config that tells the builder to use the `ecr-login` credential helper for that registry. It holds
   no password: the builder fetches a fresh ECR token on every build, so nothing expires.
```
{"auths": {}, "credHelpers": {"123456789012.dkr.ecr.us-east-1.amazonaws.com": "ecr-login"}}
```

3. Store it as an image pull secret:
```
flyte create secret --type image_pull ecr-cross-account --from-file config.json
```
If your organization has more than one cluster pool, add `--cluster-pool <pool>`.

`from_base()` doesn't take a secret. Pass it as `registry_secret` on `clone()`, which you already need in order to
add layers on top of the base image. Leave `registry` unset to push the built image to the default registry.
"""

image = (
    Image.from_base("123456789012.dkr.ecr.us-east-1.amazonaws.com/my-base-image:latest")
    .clone(name="my-image", registry_secret="ecr-cross-account", extendable=True)
    .with_pip_packages("flyte")
)

env = flyte.TaskEnvironment(name="ecr-cross-account", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(t1, data="world")
    print(run.name)
    print(run.url)
