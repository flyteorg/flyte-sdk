import asyncio
import os

from basics.hello import main

import flyte


async def integration_tests():
    tests = [
        flyte.run.aio(main, x_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ]
    runs = await asyncio.gather(*tests)
    for r in runs:
        print(r.url)
        r.wait()


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
    asyncio.run(integration_tests())
