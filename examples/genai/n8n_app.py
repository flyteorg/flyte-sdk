import flyte
import flyte.app

image = (
    flyte.Image.from_base("node:24-slim")
    .clone(name="n8n-app-image")
    .with_pip_packages("flyte==2.0.0b52")
    .with_apt_packages("ca-certificates", "curl", "gnupg", "npm")
    .with_commands(
        [
            "node --version",
            "npm --version",
            "npm install -g n8n@2.4.8"
        ],
    )
)


n8n_app = flyte.app.AppEnvironment(
    name="n8n-app",
    image=image,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    port=5678,
    args=["n8n", "start"],
)

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(n8n_app)
    print(app.url)
