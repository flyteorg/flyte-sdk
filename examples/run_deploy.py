import flyte

if __name__ == "__main__":
    flyte.init_from_config()
    task = flyte.remote.Task.get("named_fanout.fanout", auto_version="latest")
    result = flyte.run(task, n1=2, n2=3, n3=4)
    print(f"Result: {result}")