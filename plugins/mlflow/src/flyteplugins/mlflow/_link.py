from dataclasses import dataclass

from flyte import Link


@dataclass
class Mlflow(Link):
    """MLflow UI link for Flyte tasks.

    Resolves the link URL from one of two sources (in priority order):

    1. **Explicit link** — set at definition or override time::

           @env.task(links=[Mlflow(link="https://mlflow.example.com/...")])

           task.override(links=[Mlflow(link="https://...")])()

    2. **Context link** — auto-generated from `link_host` (and optional
       `link_template`) set via `mlflow_config()`. Propagates to child
       tasks that share or nest under the parent's run. Cleared when a task
       creates an independent run (`run_mode="new"`). For nested runs
       (`run_mode="nested"`), the parent link is kept and the link name
       is automatically set to "MLflow (parent)".
    """

    name: str = "MLflow"
    link: str = ""
    _decorator_run_mode: str = ""

    def get_link(
        self,
        run_name: str,
        project: str,
        domain: str,
        context: dict[str, str],
        parent_action_name: str,
        action_name: str,
        pod_name: str,
        **kwargs,
    ) -> str:
        if self.link:
            return self.link

        # Don't show inherited parent link when task creates its own run.
        # Check decorator-level run_mode (set at decoration time) and
        # context-level run_mode (set via mlflow_config context manager).
        run_mode = self._decorator_run_mode or (context.get("mlflow_run_mode") if context else None)
        if run_mode == "new":
            return ""

        return context.get("_mlflow_link", "") if context else ""
