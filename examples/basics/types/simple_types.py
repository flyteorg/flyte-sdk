import flyte
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

env = flyte.TaskEnvironment(name="inputs_simple_types")


@env.task
def main(str: str, int: int, float: float, bool: bool, start_time: datetime, duration: timedelta) -> str:
    """Process simple types and complex dataclass with all supported subtypes"""
    result = f"Hello, simple types: str={str}, int={int}, float={float}, bool={bool}"
    result += f"\nTime: {start_time}, Duration: {duration}"
    return result


if __name__ == "__main__":
    flyte.init_from_config()

    r = flyte.run(main, str="World", int=42, float=3.14, bool=True, start_time=datetime.now(), duration=timedelta(hours=1))
    print(r.name)
    print(r.url)
    r.wait()