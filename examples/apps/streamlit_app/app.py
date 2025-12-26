import flyte
import flyte.app

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "streamlit==1.41.1", "pandas==2.2.3", "numpy==2.2.3"
)


app_env = flyte.app.AppEnvironment(
    name="streamlit-custom-code",
    image=image,
    args="streamlit run main.py --server.port 8080",
    port=8080,
    include=["main.py", "utils.py"],
    resources=flyte.Resources(cpu="1", memory="1Gi"),
)


if __name__ == "__main__":
    flyte.init_from_config()
    d = flyte.deploy(app_env)
    print(d[0].table_repr())
