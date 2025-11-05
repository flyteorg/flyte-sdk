import enum

import flyte

env = flyte.TaskEnvironment(name="inputs_enum_types")


class Status(enum.Enum):
    """Example enum with string values"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@env.task
def main(status: Status) -> str:
    """Process enum type input"""
    result = f"Status received: {status.value}"
    result += f"\nStatus name: {status.name}"
    return result


if __name__ == "__main__":
    flyte.init_from_config()

    r = flyte.run(main, status=Status.COMPLETED)
    print(r.name)
    print(r.url)
    r.wait()

