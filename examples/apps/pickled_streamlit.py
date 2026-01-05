"""A pickled Streamlit app example."""

from pathlib import Path

import flyte
import flyte.app


streamlit_script = """
import streamlit as st

st.set_page_config(page_title="Simple Streamlit App", page_icon="🚀")

st.title("Hello from Streamlit!")
st.write("This is a simple single-script Streamlit app.")

name = st.text_input("What's your name?", "World")
st.write(f"Hello, {name}!")

if st.button("Click me!"):
    st.balloons()
    st.success("Button clicked!")
"""


file_name = Path(__file__).name
app_env = flyte.app.AppEnvironment(
    name="streamlit-pickled",
    image=flyte.Image.from_debian_base().with_pip_packages("streamlit==1.41.1"),
    port=8080,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
)


@app_env.server
def streamlit_app_server():
    import subprocess

    with open("./__streamlit_app__.py", "w") as f:
        f.write(streamlit_script)

    subprocess.run(["streamlit", "run", "./__streamlit_app__.py", "--server.port", "8080"], check=False)


if __name__ == "__main__":
    import logging

    flyte.init_from_config(
        root_dir=Path(__file__).parent,
        log_level=logging.DEBUG,
    )
    app = flyte.with_servecontext(interactive_mode=True).serve(app_env)
    print(f"App URL: {app.url}")
