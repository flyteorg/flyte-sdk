import flyte
from flyte.app import AppEnvironment

env = AppEnvironment(
    name="vue-union-app",
    image=(
        flyte.Image.from_debian_base()
        .with_apt_packages("nodejs", "npm")
    ),
    args="sh start.sh",
    port=8080,
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    include=["App.vue", "main.js", "index.html", "package.json", "vite.config.js", "start.sh"],
    requires_auth=False,
)

if __name__ == "__main__":
    flyte.init_from_config()
    deployment = flyte.serve(env)
    print(f"App deployed at: {deployment.url}")
