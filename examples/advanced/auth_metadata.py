import flyte
import flyte.remote as remote

flyte.init_passthrough("endpoint")

with remote.auth_metadata(("key1", "value1")):
    print(remote.Run.listall(limit=2))
