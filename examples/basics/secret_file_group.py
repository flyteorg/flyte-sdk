import flyte
import pathlib

GROUP = "arn:aws:secretsmanager:us-east-2:XXXXXXXXXX:secret"
KEY = "V2_SECRET_TEST"
SECRET_PATH = "/etc/flyte/secrets"

env = flyte.TaskEnvironment(
    "secret-fun",
    secrets=flyte.Secret(group=GROUP, key=KEY, mount=pathlib.Path(SECRET_PATH)),
)


@env.task
def main() -> str:
    return pathlib.Path(f"{SECRET_PATH}/{GROUP}/{KEY.lower()}").read_text()


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
