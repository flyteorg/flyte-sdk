from pydantic import BaseModel

import flyte

env = flyte.TaskEnvironment(name="torch_env")


class MyInput(BaseModel):
    x: int
    y: int


class MyData(BaseModel):
    some_string: str
    some_float: float
    some_bool: bool


@env.task
async def torch_task(v: MyInput, z: int = 10) -> MyData:
    return MyData(some_string="v", some_float=float(v.x + v.y + z), some_bool=True)
