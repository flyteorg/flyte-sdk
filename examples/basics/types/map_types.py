import flyte

env = flyte.TaskEnvironment(name="inputs_map_types")


@env.task
def main(data: dict[str, int]) -> str:
    """Process map/dictionary type input"""
    result = f"Received map with {len(data)} entries:\n"
    for key, value in data.items():
        result += f"  {key}: {value}\n"
    result += f"\nTotal sum: {sum(data.values())}"
    return result


if __name__ == "__main__":
    flyte.init_from_config()

    test_map = {"apple": 5, "banana": 3, "orange": 7, "grape": 2}
    r = flyte.run(main, data=test_map)
    print(r.name)
    print(r.url)
    r.wait()

