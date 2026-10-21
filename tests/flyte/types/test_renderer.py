import json
from unittest.mock import patch

from flyte.types._renderer import PythonDependencyRenderer


def _fake_check_output(cmd):
    if "list" in cmd:
        return json.dumps([{"name": "flyte", "version": "2.0.0"}]).encode()
    return b"flyte==2.0.0\n"


def test_python_dependency_renderer_lists_packages():
    with patch("subprocess.check_output", side_effect=_fake_check_output):
        html = PythonDependencyRenderer().to_html()
    assert "<td>flyte</td>" in html
    assert "flyte==2.0.0" in html


def test_python_dependency_renderer_reports_pip_failure():
    with patch("subprocess.check_output", side_effect=OSError("no pip")):
        html = PythonDependencyRenderer().to_html()
    assert html == "Error occurred while fetching installed packages."
