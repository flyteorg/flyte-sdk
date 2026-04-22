from pathlib import Path


def render() -> str:
    template = (Path(__file__).parent / "report_template.html").read_text()
    return template.format(title="Hello", body="Rendered from a bundled template.")
