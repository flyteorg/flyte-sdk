"""
Probe: when a raw container task takes list[File] / dict[str, File] inputs,
does Flyte CoPilot
  (a) localize every nested blob under /var/inputs/, and
  (b) rewrite the URIs inside /var/inputs/inputs.json to those local paths?

The container just dumps the inputs directory listing and inputs.json so we
can read it from the streamed logs.
"""

import tempfile

import flyte
from flyte.extras import ContainerTask
from flyte.io import File

probe = ContainerTask(
    name="probe_inputs_json",
    image=flyte.Image.from_debian_base().with_apt_packages("jq"),
    inputs={
        "files_list": list[File],
        "files_map": dict[str, File],
        "flags": list[str],
        "extras": dict[str, str],
    },
    outputs={"inputs_json": File},
    metadata_format="JSON",
    command=[
        "/bin/sh",
        "-c",
        r"""
set -eu
echo "==================== /var/inputs LISTING ===================="
ls -laR /var/inputs/ || true
echo
echo "==================== inputs.json (pretty) ===================="
if [ -f /var/inputs/inputs.json ]; then
  jq . /var/inputs/inputs.json || cat /var/inputs/inputs.json
  cp /var/inputs/inputs.json /var/outputs/inputs_json
else
  echo "no inputs.json found; listing what is there:"
  ls -la /var/inputs/
  # still satisfy the output so the task can complete
  echo "{}" > /var/outputs/inputs_json
fi
echo
echo "==================== blob URIs from inputs.json ============="
if command -v jq >/dev/null 2>&1 && [ -f /var/inputs/inputs.json ]; then
  echo "-- files_list URIs --"
  jq -r '..|.uri? // empty' /var/inputs/inputs.json || true
fi
echo
echo "==================== read each list file =============="
for f in /var/inputs/files_list/*; do
  [ -f "$f" ] || continue
  echo "FILE: $f"
  head -c 200 "$f"
  echo
done
echo "==================== read each map file =============="
for f in /var/inputs/files_map/*; do
  [ -f "$f" ] || continue
  echo "FILE: $f"
  head -c 200 "$f"
  echo
done
echo "==================== build argv from inputs.json (jq) =============="
ARGS=$(jq -r '.flags[]' /var/inputs/inputs.json | tr '\n' ' ')
KW=$(jq -r '.extras | to_entries[] | "\(.key) \(.value)"' /var/inputs/inputs.json | tr '\n' ' ')
BAMS=$(jq -r '.files_list[]' /var/inputs/inputs.json | tr '\n' ' ')
echo "Composed argv:  some_tool $ARGS $KW $BAMS"
""",
    ],
)

env = flyte.TaskEnvironment(
    name="probe_env",
    depends_on=[flyte.TaskEnvironment.from_task("probe_inner", probe)],
)


@env.task
async def main() -> File:
    # Generate two tiny files locally and upload them as File inputs.
    paths = []
    for i, body in enumerate(["hello from file 0\n", "hello from file 1\n"]):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_{i}.txt", delete=False
        ) as f:
            f.write(body)
            paths.append(f.name)

    files_list = [await File.from_local(p) for p in paths]
    files_map = {
        "alpha": await File.from_local(paths[0]),
        "beta": await File.from_local(paths[1]),
    }

    return await probe(
        files_list=files_list,
        files_map=files_map,
        flags=["--threads", "8", "-v"],
        extras={"--memory": "4G", "--out": "/tmp/out"},
    )


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.with_runcontext(mode="remote").run(main)
    print(r.url)
