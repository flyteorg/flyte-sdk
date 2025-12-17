"""A single-script Streamlit app example."""

import sys
from pathlib import Path

import streamlit as st

import flyte
import flyte.app


def main():
    st.set_page_config(page_title="Simple Streamlit App", page_icon="🚀")

    st.title("Hello from Streamlit!")
    st.write("This is a simple single-script Streamlit app.")

    name = st.text_input("What's your name?", "World")
    st.write(f"Hello, {name}!")

    if st.button("Click me!"):
        st.balloons()
        st.success("Button clicked!")


file_name = Path(__file__).name
app_env = flyte.app.AppEnvironment(
    name="streamlit-single-script",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("streamlit==1.41.1"),
    args=[
        "streamlit",
        "run",
        file_name,
        "--server.port",
        "8080",
        "--",
        "--server",
    ],
    port=8080,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    requires_auth=False,
)

if __name__ == "__main__":
    import logging
    import sys

    if "--server" in sys.argv:
        main()
    else:
        flyte.init_from_config(
            root_dir=Path(__file__).parent,
            log_level=logging.DEBUG,
        )
        app = flyte.serve(app_env)
        print(f"App URL: {app.url}")
