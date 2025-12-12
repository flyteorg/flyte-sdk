import flyte

# TaskEnvironments provide a simple way of grouping configuration used by tasks (more later).
env = flyte.TaskEnvironment(name="hello_world", resources=flyte.Resources(memory="250Mi"))


# use TaskEnvironments to define tasks, which are regular Python functions.
@env.task
def main(name: str) -> str:
    message = f"Hello {name}!"
    print(message)
    return message


if __name__ == "__main__":
    flyte.init_from_config()  # establish remote connection from within your script.
    run = flyte.run(main, name="World")  # run remotely inline and pass data.

    # print various attributes of the run.
    print(run.name)
    print(run.url)

    run.wait()  # stream the logs from the root action to the terminal.
