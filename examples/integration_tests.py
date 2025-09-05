import os

from basics.hello import main

import flyte

env = flyte.TaskEnvironment(name="integration_tests")


@env.task
async def integration_tests() -> None:
    main([1, 2, 3])


if __name__ == "__main__":
    flyte.init(
        endpoint="dns:///playground.canary.unionai.cloud",
        auth_type="ClientSecret",
        client_id="flyte-sdk-ci",
        client_credentials_secret=os.getenv("FLYTE_SDK_CI_TOKEN"),
        insecure=False,
        image_builder="remote",
        project="flyte-sdk",
        domain="development",
    )
    run = flyte.run(integration_tests)

    print(run.name)
    print(run.url)
    run.wait(run)
