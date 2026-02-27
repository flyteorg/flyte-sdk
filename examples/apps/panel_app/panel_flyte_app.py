"""Panel + Textual app served as a Flyte AppEnvironment.

This example demonstrates how to serve a Panel app with an embedded Textual
application using the @app_env.server decorator pattern. The app has:
- Left side: A read-only code editor with a "Play" button to run the code locally
- Right side: The ExploreTUIApp for browsing Flyte entities

Create the Gemini API key:
    flyte create secret GOOGLE_GEMINI_API_KEY

Usage (CLI):
    flyte serve --local examples/apps/panel_app/panel_flyte_app.py app_env
"""

import importlib.util
import os
from pathlib import Path

from textual.app import App

import flyte
from flyte.app import AppEnvironment, Domain, Scaling
from flyte.cli._tui._explore import ExploreScreen

app_env = AppEnvironment(
    name="panel-textual-app-test-1",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "panel",
        "textual",
        "scikit-learn",
        "torch",
        "torchvision",
        "pandas",
        "pyarrow",
        "joblib",
        "langgraph",
        "langchain-core",
        "langchain-google-genai",
    ),
    port=8080,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    scaling=Scaling(
        replicas=(0, 5),
        metric=Scaling.RequestRate(10),
        scaledown_after=300,
    ),
    secrets=[flyte.Secret(key="GOOGLE_GEMINI_API_KEY")],
    domain=Domain(subdomain="flyte2intro"),
    include=[
        "explore.tcss",
        "sample_hello_world.py",
        "sample_async.py",
        "sample_caching_and_retries.py",
        "sample_pbj_agent.py",
        "sample_distributed_random_forest.py",
        "sample_mnist_training.py",
        "sample_langgraph_gemini_agent.py",
    ],
    requires_auth=False,
)

EXAMPLES_DIR = Path(__file__).parent


def _example_script_path(script_name: str) -> Path:
    return EXAMPLES_DIR / script_name


def _load_example_script(script_name: str) -> str:
    return _example_script_path(script_name).read_text()


def _load_example_module(script_name: str):
    script_path = _example_script_path(script_name)
    module_name = f"panel_example_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_explore_css() -> str:
    """Load explore.tcss from multiple possible locations."""
    possible_paths = [
        Path(__file__).parent / "explore.tcss",  # Local development
        Path.cwd() / "explore.tcss",  # Remote: file copied to working dir
        Path("/app/explore.tcss"),  # Remote: common container path
    ]
    for path in possible_paths:
        if path.exists():
            return path.read_text()
    return ""  # Fallback to empty CSS if not found


class CustomExploreScreen(ExploreScreen):
    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Disable the quit and palette actions."""
        if action in ("quit_app", "command_palette"):
            return False
        return True


class CustomExploreTUIApp(App[None]):
    """Custom app that inherits from App directly."""

    TITLE = "Runs list"
    CSS = _load_explore_css()
    ENABLE_COMMAND_PALETTE = False

    def on_mount(self) -> None:
        self.push_screen(CustomExploreScreen())


EXAMPLES = {
    "Hello World": {
        "script": "sample_hello_world.py",
        "description": "A simple example showing tasks, parallel mapping, conditionals, and basic orchestration.",
        "run_kwargs": {"x_list": list(range(1, 11))},
    },
    "Async Python": {
        "script": "sample_async.py",
        "description": "An example showing a map-reduce pattern with asynchronous tasks and parallel execution.",
        "run_kwargs": {"count": 10},
    },
    "Caching and Retries": {
        "script": "sample_caching_and_retries.py",
        "description": "An example showing caching and retries.",
        "run_kwargs": {"user_id": 123},
    },
    "PBJ Sandwich Dummy Agent": {
        "script": "sample_pbj_agent.py",
        "description": "An async agent that plans and executes steps with dependencies to make a sandwich.",
        "run_kwargs": {"goal": "Make a peanut butter and jelly sandwich"},
    },
    "LangGraph Gemini Agent": {
        "script": "sample_langgraph_gemini_agent.py",
        "description": "A LangGraph ReAct-style agent using Google Gemini tool calling.",
        "run_kwargs": {
            "prompt": "What is the weather forecast in Berlin tomorrow, and should I bring a jacket?"
        },
        "env_vars": ["GOOGLE_GEMINI_API_KEY"],
    },
    "Distributed Random Forest": {
        "script": "sample_distributed_random_forest.py",
        "description": "A simple distributed random forest training implementation.",
        "run_kwargs": {"n_estimators": 8},
    },
    "MNIST Training": {
        "script": "sample_mnist_training.py",
        "description": "Train a simple classifier on MNIST-style handwritten digits.",
        "run_kwargs": {"sample_size": 1000, "test_size": 0.2},
    },
}


def create_panel_app():
    import panel as pn

    pn.extension(
        "codeeditor",
        "terminal",
        reconnect=True,
        notifications=True,
        disconnect_notification=True,
        ready_notification=True,
    )

    # Example selector tabs
    example_selector = pn.widgets.Select(
        name="Select an example",
        groups={
            "Basics": ["Hello World", "Async Python", "Caching and Retries"],
            "Agents": ["PBJ Sandwich Dummy Agent", "LangGraph Gemini Agent"],
            "ML": ["Distributed Random Forest", "MNIST Training"],
        },
        value="Hello World",
        stylesheets=[
            ":host .bk-input { background-color: #171020 !important; color: #f7f5fd !important; font-size: 14px "
            "!important; border: 1px solid #7652a2 !important; background-image: url('data:image/svg+xml,%3Csvg "
            "xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 12 12%27%3E%3Cpath d=%27M2 4l4 4 4-4%27 "
            "fill=%27none%27 stroke=%27%237652a2%27 stroke-width=%271.5%27 stroke-linecap=%27round%27 "
            "stroke-linejoin=%27round%27/%3E%3C/svg%3E') !important; background-repeat: no-repeat !important; "
            "background-position: right 10px center !important; background-size: 12px 12px !important; }",
            ":host label { color: white !important; margin-bottom: 5px !important; }",
        ],
    )

    example_description = pn.pane.Markdown(
        f"*{EXAMPLES['Hello World']['description']}*",
        styles={"color": "#d4d4d4", "font-size": "14px"},
    )

    code_editor = pn.widgets.CodeEditor(
        value=_load_example_script(EXAMPLES["Hello World"]["script"]),
        language="python",
        theme="dracula",
        readonly=True,
        height=425,
        sizing_mode="stretch_width",
        stylesheets=[
            ":host .ace_editor { font-size: 14px !important; }",
            ":host .ace_scrollbar-v { width: 6px !important; }",
            ":host .ace_scrollbar-h { height: 6px !important; }",
            ":host .ace_scrollbar-v::-webkit-scrollbar { width: 6px !important; }",
            ":host .ace_scrollbar-h::-webkit-scrollbar { height: 6px !important; }",
            ":host .ace_scrollbar::-webkit-scrollbar-track { background: transparent !important; }",
            ":host .ace_scrollbar::-webkit-scrollbar-thumb { background: #444 !important; "
            "border-radius: 3px !important; }",
            ":host .ace_scrollbar::-webkit-scrollbar-thumb:hover { background: #555 !important; }",
        ],
    )

    def on_example_change(event):
        selected = event.new
        example = EXAMPLES[selected]
        code_editor.value = _load_example_script(example["script"])
        example_description.object = f"*{example['description']}*"

    example_selector.param.watch(on_example_change, "value")

    output_area = pn.pane.Str(
        "Run the code to see the results here.",
        styles={
            "background": "#1e1e1e",
            "color": "#d4d4d4",
            "padding": "10px",
            "text-align": "left",
            "font-size": "14px",
            "white-space": "pre",
            "overflow-x": "auto",
        },
        sizing_mode="stretch_width",
        stylesheets=[
            ":host::-webkit-scrollbar { width: 6px !important; height: 6px !important; }",
            ":host::-webkit-scrollbar-track { background: transparent !important; }",
            ":host::-webkit-scrollbar-thumb { background: #444 !important; border-radius: 3px !important; }",
            ":host::-webkit-scrollbar-thumb:hover { background: #555 !important; }",
        ],
    )

    flyte_tui_app = CustomExploreTUIApp()
    textual_pane = pn.pane.Textual(
        flyte_tui_app,
        sizing_mode="stretch_both",
        min_height=800,
        stylesheets=[
            ":host { padding: 0 !important; margin: 0 !important; }",
            ":host .xterm-viewport { overflow-y: hidden !important; width: 100% !important; }",
            ":host .xterm-screen { width: 100% !important; }",
            ":host .xterm-viewport::-webkit-scrollbar { display: none !important; width: 0 !important; }",
            ":host .xterm { overflow: hidden !important; }",
        ],
        styles={"padding": "0", "margin": "0", "background": "#171020", "overflow": "hidden"},
    )

    def refresh_textual_app():
        import threading

        def _do_refresh():
            import time

            time.sleep(0.5)
            if flyte_tui_app._running:
                flyte_tui_app.call_from_thread(lambda: flyte_tui_app.screen.query_one("#runs-table").populate())

        thread = threading.Thread(target=_do_refresh, daemon=True)
        thread.start()

    def run_code(event):
        output_area.object = "▶️ Running code locally...\n"
        try:
            selected_example = example_selector.value
            selected_config = EXAMPLES[selected_example]
            module = _load_example_module(selected_config["script"])

            if not hasattr(module, "main"):
                output_area.object = "Error: Could not find 'main' task in the code."
                return

            flyte.init(local_persistence=True)
            env_vars = None
            if "env_vars" in selected_config:
                env_vars = {}
                for env_var in selected_config["env_vars"]:
                    env_vars[env_var] = os.getenv(env_var)

            run_kwargs = selected_config["run_kwargs"]
            run = flyte.with_runcontext(mode="local", env_vars=env_vars).run(
                module.main,
                **run_kwargs,
            )
            result = run.outputs()
            output_area.object = (
                f"✅ Run completed!\nResult: {result}\nSee the 'Explore Runs' pane on the right for more details."
            )

            # Refresh the Textual app to show the new run
            refresh_textual_app()
        except Exception as e:
            output_area.object = f"Error: {e}"

    play_button = pn.widgets.Button(
        name="▶ Run",
        button_type="primary",
        width=150,
        stylesheets=[
            ":host .bk-btn { background-color: #7652a2 !important; font-size: 14px !important; "
            "border: none !important; line-height: 28px !important;}",
        ],
    )
    play_button.on_click(run_code)

    def format_run_kwargs(kwargs: dict) -> str:
        return ", ".join(f"{k}={v!r}" for k, v in kwargs.items())

    run_kwargs_display = pn.pane.Str(
        f"main({format_run_kwargs(EXAMPLES['Hello World']['run_kwargs'])})",
        sizing_mode="stretch_width",
        styles={
            "background": "#1e1e1e",
            "color": "#d4d4d4",
            "padding": "10px",
            "text-align": "left",
            "font-size": "14px",
            "white-space": "pre",
            "overflow-x": "auto",
        },
        stylesheets=[
            ":host::-webkit-scrollbar { width: 6px !important; height: 6px !important; }",
            ":host::-webkit-scrollbar-track { background: transparent !important; }",
            ":host::-webkit-scrollbar-thumb { background: #444 !important; border-radius: 3px !important; }",
            ":host::-webkit-scrollbar-thumb:hover { background: #555 !important; }",
        ],
    )

    def update_run_kwargs_display(event):
        selected = event.new
        run_kwargs_display.object = f"main({format_run_kwargs(EXAMPLES[selected]['run_kwargs'])})"

    example_selector.param.watch(update_run_kwargs_display, "value")

    button_row = pn.Row(
        run_kwargs_display,
        play_button,
        sizing_mode="stretch_width",
        align="end",
        styles={"margin-top": "2px", "margin-bottom": "2px"},
    )

    left_panel = pn.Column(
        pn.pane.Markdown(
            "## Introduction to Flyte 2",
            styles={"color": "white", "font-size": "18px"},
            stylesheets=[":host h2 { margin-top: 0 !important; margin-bottom: 5px !important; }"],
        ),
        pn.pane.Markdown(
            "<a href='https://www.union.ai/docs/v2/flyte/' target='_blank'>Flyte 2</a> is a type-safe, "
            "distributed orchestrator for agents, AI, ML, and data workloads.<br>This demo walks you through how "
            "Flyte 2 works with code you can run in the browser without having to install anything. Select an example "
            "below and run it to see it in action.",
            styles={"color": "white", "font-size": "16px"},
            stylesheets=[
                ":host p { margin-top: 0 !important; margin-bottom: 5px !important; }",
                ":host a, :host a:visited { color: #a082c4 !important; }",
            ],
        ),
        example_selector,
        example_description,
        code_editor,
        pn.pane.Markdown(
            "#### Input",
            styles={"color": "white", "font-size": "16px"},
            stylesheets=[":host h4 { margin-top: 0 !important; margin-bottom: 0px !important; }"],
        ),
        button_row,
        pn.pane.Markdown(
            "#### Output",
            styles={"color": "white", "font-size": "16px"},
            stylesheets=[":host h4 { margin-top: 0 !important; margin-bottom: 0px !important; }"],
        ),
        output_area,
        sizing_mode="stretch_both",
        min_height=800,
        styles={"background": "#2d2d2d", "padding": "10px", "height": "100vh"},
    )

    # Toggle button for maximizing/minimizing the right panel
    is_maximized = pn.widgets.Toggle(
        name="⛶",
        value=False,
        button_type="default",
        width=40,
        stylesheets=[
            ":host .bk-btn { background-color: #7652a2 !important; color: #f7f5fd !important; "
            "font-size: 16px !important; border: none !important; padding: 2px 8px !important; "
            "margin-right: 10px !important; }"
        ],
    )

    right_panel = pn.Column(
        pn.Row(
            pn.Spacer(sizing_mode="stretch_width"),
            is_maximized,
            sizing_mode="stretch_width",
            styles={"background": "#2d2d2d"},
        ),
        textual_pane,
        sizing_mode="stretch_width",
        styles={"background": "#2d2d2d", "padding": "10px", "height": "100vh"},
    )

    def update_layout(maximized):
        if maximized:
            left_panel.visible = False
            is_maximized.name = "⛶"
        else:
            left_panel.visible = True
            is_maximized.name = "⛶"

    is_maximized.param.watch(lambda event: update_layout(event.new), "value")

    main_content = pn.Row(
        left_panel,
        right_panel,
        sizing_mode="stretch_width",
        min_height=800,
    )

    header = pn.Row(
        pn.pane.HTML(
            """
            <div style="display: flex; justify-content: center; align-items: center; gap: 20px; width: 100%;">
                <span style="color: #d4d4d4; font-size: 14px;">
                    Flyte 2 available now for local execution - cloud execution coming to OSS soon.
                </span>
                <a href="https://www.union.ai/try-flyte-2" target="_blank"
                   style="background-color: #171020; color: #f7f5fd; padding: 5px 10px;
                          border-radius: 5px; text-decoration: none; font-weight: bold;
                          font-size: 14px; transition: background-color 0.2s; border: 1px solid #f7f5fd;">
                    Join Flyte 2 production trial ↗
                </a>
            </div>
            """,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
        styles={
            "background": "#171020",
            "padding": "5px",
            "border-top": "1px solid #171020",
        },
    )

    layout = pn.Column(
        header,
        main_content,
        sizing_mode="stretch_both",
        styles={"height": "100vh"},
    )

    template = """
{% extends base %}

{% block contents %}
{{ embed(roots.main) }}
{% endblock %}
"""

    tmpl = pn.Template(template)
    tmpl.add_panel("main", layout)
    tmpl.add_variable(
        "app_favicon",
        "https://cdn.prod.website-files.com/63bc5f38147eb46b4951579a/63bca1b93a6aa708cb0bba32_Flyte-logo-favicon-32.png",
    )
    return tmpl


@app_env.server
def serve():
    import panel as pn

    import flyte.remote
    from flyte.app import ctx

    port = app_env.get_port().port

    _ctx = ctx()
    if _ctx.mode == "local":
        origin = app_env.endpoint.replace("http://", "")
    else:
        remote_app = flyte.remote.App.get(name=app_env.name)
        origin = remote_app.endpoint.replace("https://", "")

    # Pass the function itself, not a pre-created instance.
    # Panel will call create_panel_app() for each new user session,
    # giving each user their own independent app state.
    static_dirs = {"": str(Path(__file__).parent)}
    pn.serve(
        create_panel_app,
        title="Flyte 2 Intro",
        port=port,
        show=False,
        websocket_origin=origin,
        static_dirs=static_dirs,
    )


if __name__ == "__main__":
    import argparse
    import logging
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Serve the panel app")
    parser.add_argument("--mode", choices=["local", "remote"], default="remote", help="Serve mode")
    args = parser.parse_args()

    flyte.init_from_config(root_dir=Path(__file__).parent, log_level=logging.DEBUG)
    app_handle = flyte.with_servecontext(mode=args.mode).serve(app_env)
    print(f"Panel app is ready at {app_handle.url}")
    if args.mode == "local":
        input("Press Enter to continue...")
