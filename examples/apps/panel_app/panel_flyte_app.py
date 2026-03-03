"""Panel + Textual app served as a Flyte AppEnvironment.

This example demonstrates how to serve a Panel app with an embedded Textual
application using the @app_env.server decorator pattern. The app has:
- Left side: A read-only code editor with a "Play" button to run the code locally
- Right side: The ExploreTUIApp for browsing Flyte entities

Usage (CLI):
    flyte serve --local examples/apps/panel_app/panel_flyte_app.py app_env
"""

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
        "pandas",
        "pyarrow",
        "joblib",
    ),
    port=8080,
    resources=flyte.Resources(cpu="2", memory="1Gi"),
    scaling=Scaling(replicas=(0, 5)),
    domain=Domain(subdomain="flyte2intro"),
    include=["explore.tcss"],
    requires_auth=False,
)

SAMPLE_CODE_HELLO_WORLD = """import flyte

# TaskEnvironments provide a simple way of grouping configuration used by tasks
env = flyte.TaskEnvironment(
    name="hello_world",
    resources=flyte.Resources(memory="250Mi"),
)


# use TaskEnvironments to define tasks, which are regular Python functions.
@env.task
def fn(x: int) -> int:  # type annotations are recommended.
    slope, intercept = 2, 5
    return slope * x + intercept


# tasks can also call other tasks, which will be manifested in different containers.
@env.task
def main(x_list: list[int]) -> float:
    x_len = len(x_list)
    if x_len < 10:
        raise ValueError(
            f"x_list doesn't have a larger enough sample size, found: {x_len}"
        )

    y_list = list(flyte.map(fn, x_list))  # flyte.map is like Python map, but runs in parallel.
    y_mean = sum(y_list) / len(y_list)
    return y_mean
"""

SAMPLE_CODE_PBJ_AGENT = """import asyncio
from typing import Any, Callable, Dict, List, Union

import flyte

agent_env = flyte.TaskEnvironment(
    "agent",
    resources=flyte.Resources(memory="250Mi"),
)


# --- Dummy PBJ agent that creates a plan ---
@agent_env.task
async def get_plan(goal: str) -> List[Dict[str, Union[str, List[str]]]]:
    # Each step has a name, function ID, and dependencies
    return [
        {"id": "get_bread", "deps": []},
        {"id": "get_peanut_butter", "deps": []},
        {"id": "get_jelly", "deps": []},
        {"id": "spread_peanut_butter", "deps": ["get_bread", "get_peanut_butter"]},
        {"id": "spread_jelly", "deps": ["get_bread", "get_jelly"]},
        {"id": "assemble_sandwich", "deps": ["spread_peanut_butter", "spread_jelly"]},
        {"id": "eat", "deps": ["assemble_sandwich"]},
    ]


# --- Step function definitions ---
@agent_env.task
async def get_bread(context: Dict[str, str]) -> str:
    return "bread"


@agent_env.task
async def get_peanut_butter(context: Dict[str, str]) -> str:
    return "peanut butter"


@agent_env.task
async def get_jelly(context: Dict[str, str]) -> str:
    return "jelly"


@agent_env.task
async def spread_peanut_butter(context: Dict[str, str]) -> str:
    return f"{context['get_bread']} with {context['get_peanut_butter']}"


@agent_env.task
async def spread_jelly(context: Dict[str, str]) -> str:
    return f"{context['get_bread']} with {context['get_jelly']}"


@agent_env.task
async def assemble_sandwich(context: Dict[str, str]) -> str:
    return f"{context['spread_peanut_butter']} and {context['spread_jelly']} combined"


@agent_env.task
async def eat(context: Dict[str, Any]) -> str:
    return f"Ate: {context['assemble_sandwich']} 😋"


# --- Registry of step functions ---
STEP_FUNCTIONS: Dict[str, Any | Callable[[Dict[str, Any]], Any]] = {
    "get_bread": get_bread,
    "get_peanut_butter": get_peanut_butter,
    "get_jelly": get_jelly,
    "spread_peanut_butter": spread_peanut_butter,
    "spread_jelly": spread_jelly,
    "assemble_sandwich": assemble_sandwich,
    "eat": eat,
}


# --- Executor that respects dependencies ---
@agent_env.task
async def execute_plan(plan: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, str]:
    step_funcs = STEP_FUNCTIONS
    results = {}
    remaining = {step["id"]: step for step in plan}

    i = 0
    while remaining:
        with flyte.group(f"step-set-{i}"):
            print(f"{results}")
            # Find all steps that are ready to run (no unmet dependencies)
            ready = [step_id for step_id, step in remaining.items() if all(dep in results for dep in step["deps"])]

            # Run all ready steps concurrently
            tasks = {step_id: asyncio.create_task(step_funcs[step_id](results)) for step_id in ready}

            for step_id, task in tasks.items():
                result = await task
                print(f"✅ {step_id}: {result}")
                results[step_id] = result
                del remaining[step_id]
            i = i + 1

    return results


# --- Main async driver ---
@agent_env.task
async def main(goal: str = "Make a peanut butter and jelly sandwich") -> Dict[str, str]:
    plan = await get_plan(goal)
    print(f"📋 Plan with dependencies:\\n{plan}")
    results = await execute_plan(plan)
    print("\\n🏁 Final Result:")
    return results
"""


SAMPLE_CODE_DISTRIBUTED_RANDOM_FOREST = """import asyncio
import tempfile

import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import flyte
import flyte.errors
from flyte.io import Dir, File

env = flyte.TaskEnvironment(
    name="distributed_random_forest",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)

N_FEATURES = 10


@env.task
async def create_dataset(n_estimators: int) -> Dir:
    '''Create a synthetic dataset.'''

    temp_dir = tempfile.mkdtemp()

    for i in range(n_estimators):
        print(f"Creating dataset {i}")
        X, y = make_classification(
            n_samples=1_000,
            n_classes=2,
            n_features=N_FEATURES,
            n_informative=5,
            n_redundant=3,
            n_clusters_per_class=1,
        )
        dataset = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(N_FEATURES)])
        dataset["target"] = y
        dataset.to_parquet(f"{temp_dir}/dataset_{i}.parquet")
        del X, y, dataset

    return await Dir.from_local(temp_dir)


async def get_partition(dataset_dir: Dir, dataset_index: int) -> pd.DataFrame:
    '''Helper function to get a partition of the dataset.'''

    async for file in dataset_dir.walk():
        if file.name == f"dataset_{dataset_index}.parquet":
            local_path = await file.download()

    return pd.read_parquet(local_path)


@env.task
async def train_decision_tree(dataset_dir: Dir, dataset_index: int) -> File:
    '''Train a decision tree on a subset of the dataset.'''

    print(f"Training decision tree on partition {dataset_index}")
    dataset = await get_partition(dataset_dir, dataset_index)
    y = dataset["target"]
    X = dataset.drop(columns=["target"])
    model = DecisionTreeClassifier()
    model.fit(X, y)

    temp_dir = tempfile.mkdtemp()
    fp = f"{temp_dir}/decision_tree_{dataset_index}.joblib"
    joblib.dump(model, fp)
    return await File.from_local(fp)


async def load_decision_tree(file: File) -> DecisionTreeClassifier:
    local_path = await file.download()
    return joblib.load(local_path)


def random_forest_from_decision_trees(decision_trees: list[DecisionTreeClassifier]) -> RandomForestClassifier:
    '''Helper function that reconstitutes a random forest model from a list of decision trees.'''

    rf = RandomForestClassifier(n_estimators=len(decision_trees))
    rf.estimators_ = decision_trees
    rf.classes_ = decision_trees[0].classes_
    rf.n_classes_ = decision_trees[0].n_classes_
    rf.n_features_in_ = decision_trees[0].n_features_in_
    rf.n_outputs_ = decision_trees[0].n_outputs_
    rf.feature_names_in_ = [f"feature_{i}" for i in range(N_FEATURES)]
    return rf


@env.task
async def train_distributed_random_forest(dataset_dir: Dir, n_estimators: int) -> File:
    '''Train a distributed random forest on the dataset.

    Random forest is an ensemble of decision trees that have been trained
    on subsets of a dataset. Here we implement distributed random forest where
    the full dataset cannot be loaded into memory. We therefore load partitions
    of the data into its own task and train decision tree on each partition.

    After training, we reconstitute the random forest from the collection
    of trained decision tree models.
    '''

    decision_tree_files: list[File] = []

    with flyte.group(f"parallel-training-{n_estimators}-decision-trees"):
        for i in range(n_estimators):
            decision_tree_files.append(train_decision_tree(dataset_dir, i))

        decision_tree_files = await asyncio.gather(*decision_tree_files)

    decision_trees = await asyncio.gather(*[load_decision_tree(file) for file in decision_tree_files])

    random_forest = random_forest_from_decision_trees(decision_trees)
    temp_dir = tempfile.mkdtemp()
    fp = f"{temp_dir}/random_forest.joblib"
    joblib.dump(random_forest, fp)
    return await File.from_local(fp)


@env.task
async def evaluate_random_forest(
    random_forest: File,
    dataset_dir: Dir,
    dataset_index: int,
) -> float:
    '''Evaluate the random forest one partition of the dataset.'''

    with random_forest.open_sync() as f:
        random_forest = joblib.load(f)

    data_partition = await get_partition(dataset_dir, dataset_index)
    y = data_partition["target"]
    X = data_partition.drop(columns=["target"])

    predictions = random_forest.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy: {accuracy}")
    return accuracy


@env.task
async def main(n_estimators: int = 8) -> tuple[File, float]:
    dataset = await create_dataset(n_estimators=n_estimators)
    random_forest = await train_distributed_random_forest(dataset, n_estimators)
    accuracy = await evaluate_random_forest(random_forest, dataset, 0)
    return random_forest, accuracy
"""


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
        "code": SAMPLE_CODE_HELLO_WORLD,
        "description": "A simple example showing tasks, parallel mapping, and basic orchestration.",
        "run_kwargs": {"x_list": list(range(1, 11))},
    },
    "PBJ Sandwich Agent": {
        "code": SAMPLE_CODE_PBJ_AGENT,
        "description": "An async agent that plans and executes steps with dependencies to make a sandwich.",
        "run_kwargs": {"goal": "Make a peanut butter and jelly sandwich"},
    },
    "Distributed Random Forest": {
        "code": SAMPLE_CODE_DISTRIBUTED_RANDOM_FOREST,
        "description": "A simple distributed random forest training implementation.",
        "run_kwargs": {"n_estimators": 8},
    },
}


def create_panel_app():
    import panel as pn

    pn.extension("codeeditor", "terminal", reconnect=True)

    # Example selector tabs
    example_selector = pn.widgets.Select(
        name="Select an example",
        groups={
            "Basics": ["Hello World"],
            "Advanced": ["PBJ Sandwich Agent", "Distributed Random Forest"],
        },
        value="Hello World",
        stylesheets=[
            ":host .bk-input { background-color: #171020 !important; color: #f7f5fd !important; font-size: 14px "
            "!important; border: 1px solid #7652a2 !important; }",
            ":host label { color: white !important; margin-bottom: 5px !important; }",
        ],
    )

    example_description = pn.pane.Markdown(
        f"*{EXAMPLES['Hello World']['description']}*",
        styles={"color": "#d4d4d4", "font-size": "14px"},
    )

    code_editor = pn.widgets.CodeEditor(
        value=SAMPLE_CODE_HELLO_WORLD,
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
        code_editor.value = example["code"]
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

            time.sleep(0.25)
            if flyte_tui_app._running:
                flyte_tui_app.call_from_thread(lambda: flyte_tui_app.screen.query_one("#runs-table").populate())

        thread = threading.Thread(target=_do_refresh, daemon=True)
        thread.start()

    def run_code(event):
        output_area.object = "Running code locally...\n"
        try:
            namespace = {}
            exec(code_editor.value, namespace)

            if "main" in namespace:
                flyte.init(local_persistence=True)

                # Get the run kwargs for the selected example
                selected_example = example_selector.value
                run_kwargs = EXAMPLES[selected_example]["run_kwargs"]

                run = flyte.with_runcontext(mode="local").run(
                    namespace["main"],
                    **run_kwargs,
                )
                result = run.outputs()
                output_area.object = (
                    f"✅ Run completed!\nResult: {result}\nSee the 'Explore Runs' pane on the right for more details."
                )

                # Refresh the Textual app to show the new run
                refresh_textual_app()
            else:
                output_area.object = "Error: Could not find 'main' task in the code."
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
            "Flyte 2 is a type-safe, distributed orchestrator of agents, AI, ML, and more — "
            "in pure Python, with sync and async support. Run an example below to see it in action.",
            styles={"color": "white", "font-size": "16px"},
            stylesheets=[":host p { margin-top: 0 !important; margin-bottom: 5px !important; }"],
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
                    Try the Union engine for Flyte 2.0
                </span>
                <a href="https://www.union.ai/beta" target="_blank"
                   style="background-color: #171020; color: #f7f5fd; padding: 5px 10px;
                          border-radius: 5px; text-decoration: none; font-weight: bold;
                          font-size: 14px; transition: background-color 0.2s; border: 1px solid #f7f5fd;">
                    Register now ↗
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
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Serve the panel app")
    parser.add_argument("--mode", choices=["local", "remote"], default="remote", help="Serve mode")
    args = parser.parse_args()

    flyte.init_from_config(root_dir=Path(__file__).parent)
    app_handle = flyte.with_servecontext(mode=args.mode).serve(app_env)
    print(f"Panel app is ready at {app_handle.url}")
    if args.mode == "local":
        input("Press Enter to continue...")
