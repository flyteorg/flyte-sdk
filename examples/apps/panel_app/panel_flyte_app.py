"""Panel + Textual app served as a Flyte AppEnvironment.

This example demonstrates how to serve a Panel app with an embedded Textual
application using the @app_env.server decorator pattern. The app has:
- Left side: A read-only code editor with a "Play" button to run the code locally
- Right side: The ExploreTUIApp for browsing Flyte entities

Reo.Dev tracking uses the reodotdev npm package (no CDN). To rebuild after
editing reo_component.py or reo_init.js:
    cd examples/apps/panel_app && npm run build:reo
Requires: node, npm, esbuild (npm install in that directory).

Create the Gemini API key:
    flyte create secret GOOGLE_GEMINI_API_KEY

Create the Reo client ID:
    flyte create secret REO_CLIENT_ID

Usage (CLI):
    flyte serve --local examples/apps/panel_app/panel_flyte_app.py app_env
"""

import importlib.util
import os
import threading
import time
from pathlib import Path

import param
from panel.custom import JSComponent
from textual.app import App

import flyte
from flyte.app import AppEnvironment, Domain, Scaling
from flyte.cli._tui._explore import ExploreScreen

app_env = AppEnvironment(
    name="panel-textual-app-test-1",
    image=flyte.Image.from_debian_base().with_pip_packages(
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
    resources=flyte.Resources(cpu="1", memory="1Gi", disk="32Gi"),
    scaling=Scaling(
        replicas=(1, 5),
        metric=Scaling.RequestRate(3),
        scaledown_after=300,  # 5 minutes
    ),
    secrets=[
        flyte.Secret(key="GOOGLE_GEMINI_API_KEY"),
        flyte.Secret(key="REO_CLIENT_ID"),
    ],
    domain=Domain(subdomain="flyte2intro"),
    include=[
        "explore.tcss",
        "template.html",
        "header.html",
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


def _load_template_html() -> str:
    return (Path(__file__).parent / "template.html").read_text()


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


class ReoInitializer(JSComponent):
    """Invisible component that initializes Reo.Dev tracking via the reodotdev npm package."""

    client_id = param.String(
        doc="Reo.Dev client ID. Pass from REO_CLIENT_ID env var in production.",
    )

    _importmap = {  # noqa: RUF012
        "imports": {
            "reodotdev": "https://esm.sh/reodotdev@1.1.0",
        }
    }

    _esm = """
    /* reo_init.js - Reo.Dev initialization via npm package */
    import { loadReoScript } from "reodotdev";

    export function render({ model }) {
    const clientID = model.client_id

    const reoPromise = loadReoScript({ clientID });
    reoPromise
        .then((Reo) => {
        Reo.init({ clientID });
        console.log("Reo initialized");
        })
        .catch((error) => {
        console.error("Error loading Reo", error);
        });

    // Return minimal invisible container - Reo init is a side effect
    const el = document.createElement("div");
    el.style.cssText = "position:absolute;width:0;height:0;overflow:hidden;pointer-events:none;";
    return el;
    }
    """


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
        "description": (
            "Run the task twice to see the cached result, then run it again with the `disable cache` toggle to ignore "
            "the cache."
        ),
        "run_kwargs": {"user_id": 1234},
    },
    "PBJ Sandwich Dummy Agent": {
        "script": "sample_pbj_agent.py",
        "description": "An async agent that plans and executes steps with dependencies to make a sandwich.",
        "run_kwargs": {"goal": "Make a peanut butter and jelly sandwich"},
    },
    "LangGraph Gemini Agent": {
        "script": "sample_langgraph_gemini_agent.py",
        "description": "A LangGraph agent using Google Gemini tool calling to get the weather forecast.",
        "run_kwargs": {"prompt": "What is the weather forecast in Berlin tomorrow, and should I bring a jacket?"},
        "env_vars": ["GOOGLE_GEMINI_API_KEY"],
    },
    "Distributed Random Forest": {
        "script": "sample_distributed_random_forest.py",
        "description": "A simple distributed random forest training implementation with sc ikit-learn.",
        "run_kwargs": {"n_estimators": 8},
    },
    "MNIST Training": {
        "script": "sample_mnist_training.py",
        "description": "Train a simple classifier on MNIST-style handwritten digits with pytorch.",
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
        groups={
            "Basics": ["Hello World", "Async Python", "Caching and Retries"],
            "AI Agents": ["PBJ Sandwich Dummy Agent", "LangGraph Gemini Agent"],
            "Machine Learning": ["Distributed Random Forest", "MNIST Training"],
        },
        value="Hello World",
        stylesheets=[
            ":host .bk-input { background-color: #050310 !important; color: #f7f5fd !important; font-size: 14px "
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
        stylesheets=[":host p { margin-top: 0 !important; margin-bottom: 0 !important; }"],
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

    disable_cache_toggle = pn.widgets.Toggle(
        name="disable cache",
        value=False,
        visible=False,
        button_type="default",
        width=100,
        stylesheets=[
            ":host .bk-btn { background-color: #050310 !important; color: #f7f5fd !important; "
            "font-size: 12px !important; border: 1px solid #8C4FFF !important; line-height: 26px !important; }",
            ":host .bk-active { background-color: #8C4FFF !important; color: #f7f5fd !important; }",
        ],
    )

    def update_disable_cache_toggle(selected_example: str):
        is_caching_example = selected_example == "Caching and Retries"
        disable_cache_toggle.visible = is_caching_example
        if not is_caching_example:
            disable_cache_toggle.value = False

    def on_example_change(event):
        selected = event.new
        example = EXAMPLES[selected]
        code_editor.value = _load_example_script(example["script"])
        example_description.object = f"*{example['description']}*"
        output_area.object = "Run the code to see the results here."
        update_disable_cache_toggle(selected)

    example_selector.param.watch(on_example_change, "value")

    flyte_tui_app = CustomExploreTUIApp()
    _start_periodic_runs_clear(flyte_tui_app)

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
        styles={"padding": "0", "margin": "0", "background": "#050310", "overflow": "hidden"},
    )

    def refresh_textual_app(time_to_sleep: float = 0.5):
        import threading

        def _do_refresh():
            import time

            time.sleep(time_to_sleep)
            if flyte_tui_app._running:
                flyte_tui_app.call_from_thread(lambda: flyte_tui_app.screen.query_one("#runs-table").populate())

        thread = threading.Thread(target=_do_refresh, daemon=True)
        thread.start()

    def run_code(event):
        output_area.object = "▶️ Running code locally...\n"

        # Capture context on main thread before spawning worker
        doc = getattr(pn.state, "curdoc", None)
        selected_example = example_selector.value
        selected_config = EXAMPLES[selected_example]
        disable_cache = selected_example == "Caching and Retries" and disable_cache_toggle.value

        def update_output(text: str) -> None:
            """Schedule UI update on main thread (thread-safe)."""

            def _do_update():
                output_area.object = text

            if doc is not None:
                doc.add_next_tick_callback(_do_update)
            else:
                output_area.object = text

        def _run_in_thread() -> None:
            try:
                module = _load_example_module(selected_config["script"])
                if not hasattr(module, "main"):
                    update_output("Error: Could not find 'main' task in the code.")
                    return

                flyte.init(local_persistence=True)
                env_vars = None
                if "env_vars" in selected_config:
                    env_vars = {}
                    for env_var in selected_config["env_vars"]:
                        env_vars[env_var] = os.getenv(env_var)

                run_kwargs = selected_config["run_kwargs"]
                runcontext_kwargs = {"mode": "local", "env_vars": env_vars}
                if disable_cache:
                    update_output("🔄 Disabling run cache...\n")
                    runcontext_kwargs["disable_run_cache"] = True

                refresh_textual_app(time_to_sleep=0.1)
                run = flyte.with_runcontext(**runcontext_kwargs).run(
                    module.main,
                    **run_kwargs,
                )
                result = run.outputs()
                update_output(
                    f"✅ Run completed!\nResult: {result}\nSee the 'Explore Runs' pane on the right for more details."
                )
                refresh_textual_app()
            except Exception as e:
                update_output(f"Error: {e}")

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

    play_button = pn.widgets.Button(
        name="▶ Run",
        button_type="primary",
        width=150,
        stylesheets=[
            ":host .bk-btn { background-color: #8C4FFF !important; font-size: 14px !important; "
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
        disable_cache_toggle,
        play_button,
        sizing_mode="stretch_width",
        align="end",
        styles={"margin-top": "2px", "margin-bottom": "2px"},
        css_classes=["button-row"],
    )

    mobile_tui_toggle = pn.widgets.Button(
        name="Explore Runs",
        button_type="primary",
        sizing_mode="stretch_width",
        css_classes=["mobile-tui-toggle"],
        stylesheets=[
            ":host .bk-btn { background-color: #12052a !important; color: #f7f5fd !important; "
            "font-size: 14px !important; border: 1px solid #8C4FFF !important; line-height: 28px !important; "
            "margin-top: 10px !important; }",
            ":host { display: none !important; }",
            "@media (max-width: 1024px) { :host { display: block !important; } }",
        ],
    )

    left_panel = pn.Column(
        pn.pane.Markdown(
            "## Flyte 2 Live Demo",
            styles={"color": "white", "font-size": "18px"},
            stylesheets=[":host h2 { margin-top: 0 !important; margin-bottom: 5px !important; }"],
        ),
        pn.pane.Markdown(
            "<a href='https://www.union.ai/docs/v2/flyte/' target='_blank'>Flyte 2</a> is a type-safe, "
            "distributed orchestrator for agents, AI, ML, and data workloads. This in-browser demo lets you run "
            "real Flyte code - no installs required.<br><br>Select an example below, hit `▶ Run`, and explore your "
            "results in the TUI on the right 👉",
            styles={"color": "white", "font-size": "16px"},
            stylesheets=[
                ":host p { margin-top: 0 !important; margin-bottom: 5px !important; }",
                ":host a, :host a:visited { color: #8C4FFF !important; }",
                """
                :host code {
                    background-color: #8C4FFF !important;
                    color: #f7f5fd !important;
                    padding: 2px 6px !important;
                    margin: 2px 4px !important;
                    border-radius: 4px !important;
                    font-size: 14px !important;
                    font-weight: bold !important;
                }
                """,
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
        mobile_tui_toggle,
        sizing_mode="stretch_both",
        min_height=800,
        styles={"background": "#050310", "padding": "10px", "height": "100vh"},
        css_classes=["left-panel"],
        stylesheets=[
            "@media (max-width: 1024px) { :host { height: auto !important; } }",
        ],
    )

    # Toggle button for maximizing/minimizing the right panel
    is_maximized = pn.widgets.Toggle(
        name="⛶",
        value=False,
        button_type="default",
        width=40,
        stylesheets=[
            ":host .bk-btn { background-color: #8C4FFF !important; color: #f7f5fd !important; "
            "font-size: 16px !important; border: none !important; padding: 2px 8px !important; "
            "margin-right: 10px !important; }"
        ],
    )

    right_panel = pn.Column(
        pn.Row(
            pn.Spacer(sizing_mode="stretch_width"),
            is_maximized,
            sizing_mode="stretch_width",
            styles={"background": "#050310"},
        ),
        textual_pane,
        sizing_mode="stretch_width",
        styles={"background": "#050310", "padding": "10px", "height": "100vh"},
        css_classes=["right-panel"],
        stylesheets=[
            "@media (max-width: 1024px) { :host { display: none !important; } }",
            "@media (max-width: 1024px) { :host(.right-panel-visible) { display: flex !important; } }",
            "@media (max-width: 1024px) { :host { height: auto !important; min-height: 500px !important; } }",
        ],
    )

    def update_layout(maximized):
        if maximized:
            left_panel.visible = False
            right_panel.css_classes = ["right-panel", "right-panel-visible"]
            is_maximized.name = "⛶"
        else:
            left_panel.visible = True
            right_panel.css_classes = ["right-panel"]
            is_maximized.name = "⛶"

    is_maximized.param.watch(lambda event: update_layout(event.new), "value")
    mobile_tui_toggle.on_click(lambda event: setattr(is_maximized, "value", True))

    main_content = pn.Row(
        left_panel,
        right_panel,
        sizing_mode="stretch_width",
        min_height=800,
        css_classes=["main-content-row"],
        stylesheets=[
            "@media (max-width: 1024px) { :host { flex-direction: column !important; } }",
            "@media (max-width: 1024px) { :host > * { width: 100% !important; max-width: 100% !important; "
            "flex: 0 0 auto !important; } }",
        ],
    )

    header = pn.Row(
        pn.pane.HTML(
            (Path(__file__).parent / "header.html").read_text(),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
        styles={
            "background": "#12052a",
            "padding": "5px",
            "border-top": "1px solid #12052a",
        },
    )
    reo_client_id = os.getenv("REO_CLIENT_ID")
    reo_initializer = ReoInitializer(client_id=reo_client_id)

    layout = pn.Column(
        reo_initializer,
        header,
        main_content,
        sizing_mode="stretch_both",
        styles={"height": "100vh", "background": "#050310"},
    )

    template = _load_template_html()
    tmpl = pn.Template(template)
    tmpl.add_panel("main", layout)
    tmpl.add_variable(
        "app_favicon",
        "https://cdn.prod.website-files.com/690e2a44303093ad8549854b/69123f033bc348f79cd2d7a4_flyte-logo-32.png",
    )
    return tmpl


def _start_periodic_runs_clear(flyte_tui_app):
    """Clear runs (RunStore) and re-render the TUI table every 24 hours per session."""

    CACHE_CLEAR_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours

    def _clear_and_refresh():
        from flyte._persistence._run_store import RunStore

        RunStore.clear_sync()
        flyte_tui_app.screen.query_one("#runs-table").populate()

    def _clear_loop():
        while True:
            time.sleep(CACHE_CLEAR_INTERVAL_SECONDS)
            try:
                if flyte_tui_app._running:
                    flyte_tui_app.call_from_thread(_clear_and_refresh)
            except Exception:
                pass

    thread = threading.Thread(target=_clear_loop, daemon=True)
    thread.start()


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
        title="Flyte 2 | Live Demo",
        port=port,
        show=False,
        websocket_origin=origin,
        static_dirs=static_dirs,
        session_token_expiration=3 * 60 * 60,  # 3 hours
        unused_session_lifetime_milliseconds=60 * 60 * 1000,  # 1 hour
        check_unused_sessions_milliseconds=6 * 60 * 1000,  # 6 minutes
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
