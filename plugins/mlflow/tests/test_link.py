from flyteplugins.mlflow._link import Mlflow


class TestMlflow:
    def _get_link_kwargs(self, context=None):
        return {
            "run_name": "test-run",
            "project": "test-project",
            "domain": "development",
            "context": context or {},
            "parent_action_name": "parent",
            "action_name": "child",
            "pod_name": "pod-123",
        }

    def test_defaults(self):
        link = Mlflow()
        assert link.name == "MLflow"
        assert link.link == ""
        assert link._decorator_run_mode == ""

    def test_explicit_link_has_priority(self):
        link = Mlflow(link="https://explicit.com/run/123")
        result = link.get_link(**self._get_link_kwargs(context={"_mlflow_link": "https://context.com/run/456"}))
        assert result == "https://explicit.com/run/123"

    def test_context_link_used_when_no_explicit(self):
        link = Mlflow()
        result = link.get_link(**self._get_link_kwargs(context={"_mlflow_link": "https://context.com/run/456"}))
        assert result == "https://context.com/run/456"

    def test_empty_when_no_link_sources(self):
        link = Mlflow()
        result = link.get_link(**self._get_link_kwargs(context={}))
        assert result == ""

    def test_empty_context(self):
        link = Mlflow()
        result = link.get_link(**self._get_link_kwargs(context=None))
        assert result == ""

    def test_run_mode_new_suppresses_context_link(self):
        link = Mlflow(_decorator_run_mode="new")
        result = link.get_link(**self._get_link_kwargs(context={"_mlflow_link": "https://parent.com/run/123"}))
        assert result == ""

    def test_run_mode_new_from_context_suppresses_link(self):
        link = Mlflow()
        result = link.get_link(
            **self._get_link_kwargs(
                context={
                    "_mlflow_link": "https://parent.com/run/123",
                    "mlflow_run_mode": "new",
                }
            )
        )
        assert result == ""

    def test_run_mode_nested_keeps_parent_link(self):
        link = Mlflow(_decorator_run_mode="nested")
        result = link.get_link(**self._get_link_kwargs(context={"_mlflow_link": "https://parent.com/run/123"}))
        assert result == "https://parent.com/run/123"

    def test_run_mode_new_explicit_link_still_returned(self):
        link = Mlflow(link="https://explicit.com", _decorator_run_mode="new")
        result = link.get_link(**self._get_link_kwargs(context={"_mlflow_link": "https://parent.com/run/123"}))
        assert result == "https://explicit.com"

    def test_decorator_run_mode_takes_priority_over_context(self):
        link = Mlflow(_decorator_run_mode="nested")
        result = link.get_link(
            **self._get_link_kwargs(
                context={
                    "_mlflow_link": "https://parent.com/run/123",
                    "mlflow_run_mode": "new",
                }
            )
        )
        # decorator says nested, so parent link is kept
        assert result == "https://parent.com/run/123"
