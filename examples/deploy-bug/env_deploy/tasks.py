from env_deploy.environments import env_1, env_2


@env_1.task
def hello_1() -> str:
    return "hello_1"

@env_2.task
def hello_2() -> str:
    return "hello_2"