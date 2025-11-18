import flyte

env = flyte.TaskEnvironment("data_passing")


class MyObject:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"MyObject(name={self.name}, value={self.value})"


@env.task
async def downstream_task(data: MyObject) -> dict[str, MyObject]:
    return {data.name: data}


@env.task
async def upstream_task(data: MyObject) -> dict[str, MyObject]:
    data.value += 10
    obj = await downstream_task(data)
    return obj


if __name__ == "__main__":
    flyte.init_from_config()
    obj = MyObject(name="example", value=5)
    r = flyte.run(upstream_task, data=obj)
    print(r.url)
