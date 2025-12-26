"""A pickled Gradio app using @app_env.server decorator.

This example demonstrates how to create a Gradio app that gets pickled and served
using the @app_env.server decorator. The app environment is pickled (pkl bundle)
instead of using a tgz bundle, which is useful for interactive development.

Deploy with:
```
flyte init_from_config()
app = flyte.with_servecontext(interactive_mode=True).serve(env)
print(app.url)
```
"""

import flyte
import flyte.app

# Create an image with gradio installed
image = flyte.Image.from_debian_base().with_pip_packages("gradio==6.2.0")

# Create the AppEnvironment
env = flyte.app.AppEnvironment(
    name="pickled-gradio-app",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    parameters=[
        flyte.app.Parameter(name="title", value="Flyte Gradio App"),
        flyte.app.Parameter(name="description", value="A simple Gradio app deployed on Flyte"),
    ],
    requires_auth=False,
    port=7860,  # Gradio default port
)

# State that can be shared between startup and server
state = {}


@env.on_startup
async def app_startup(title: str, description: str):
    """Initialize the app state with parameters."""
    state["modified_title"] = f"{title} (modified)"
    state["modified_description"] = f"{description} (modified)"


@env.server
def app_server(title: str, description: str):
    """Server function that creates and launches the Gradio app.

    This function is called when the app is served. It creates a Gradio interface
    and launches it. The function receives parameters from the AppEnvironment.
    Note: This is a synchronous function because Gradio's launch() method is blocking.

    Args:
        title: The title for the Gradio app
        description: The description for the Gradio app
    """
    import gradio as gr

    # Create a simple Gradio interface
    def greet(name: str) -> str:
        """Simple greeting function."""
        return f"Hello, {name}! Welcome to {title}: {description}"

    def add_numbers(a: float, b: float) -> float:
        """Simple addition function."""
        return a + b

    # Create the Gradio interface
    # Note: We create the interface inside the server function so it can be pickled
    # along with the AppEnvironment
    demo = gr.Interface(
        fn=greet,
        inputs=gr.Textbox(label="Your Name", placeholder="Enter your name"),
        outputs=gr.Textbox(label="Greeting"),
        title=state["modified_title"],
        description=state["modified_description"],
    )

    # Add another tab for the calculator
    with demo:
        gr.Interface(
            fn=add_numbers,
            inputs=[
                gr.Number(label="First Number"),
                gr.Number(label="Second Number"),
            ],
            outputs=gr.Number(label="Sum"),
            title="Calculator",
        )

    # Launch the Gradio app
    # The app will be accessible on port 7860 (as specified in the AppEnvironment)
    demo.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,
        share=False,  # Set to True if you want a public Gradio link
    )


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    # Use interactive_mode=True to create a pkl bundle instead of tgz
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"App deployed at: {app.url}")
