import html
import pathlib
import string
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Union

from flyte._logging import logger
from flyte._tools import ipython_check
from flyte.syncify import syncify

if TYPE_CHECKING:
    from IPython.core.display import HTML

_MAIN_TAB_NAME = "main"


@dataclass
class Tab:
    name: str
    content: List[str] = field(default_factory=list, init=False)

    def log(self, content: str):
        """
        Add content to the tab.
        The content should be a valid HTML string, but not a complete HTML document, as it will be inserted into a div.

        :param content: The content to add.
        """
        self.content.append(content)

    def replace(self, content: str):
        """
        Replace the content of the tab.
        The content should be a valid HTML string, but not a complete HTML document, as it will be inserted into a div.

        :param content: The content to replace.
        """
        self.content = [content]

    def get_html(self) -> str:
        """
        Get the HTML representation of the tab.

        :return: The HTML representation of the tab.
        """
        return "\n".join(self.content)


@dataclass
class Report:
    name: str
    tabs: Dict[str, Tab] = field(default_factory=dict)
    template_path: pathlib.Path = field(default_factory=lambda: pathlib.Path(__file__).parent / "_template.html")

    def __post_init__(self):
        self.tabs[_MAIN_TAB_NAME] = Tab(_MAIN_TAB_NAME)

    def get_tab(self, name: str, create_if_missing: bool = True) -> Tab:
        """
        Get a tab by name. If the tab does not exist, create it.

        :param name: The name of the tab.
        :param create_if_missing: Whether to create the tab if it does not exist.
        :return: The tab.
        """
        if name not in self.tabs:
            if create_if_missing:
                self.tabs[name] = Tab(name)
            else:
                raise ValueError(f"Tab {name} does not exist.")
        return self.tabs[name]

    def get_final_report(self) -> Union[str, "HTML"]:
        """
        Get the final report as a string.

        :return: The final report.
        """
        tabs = {n: t.get_html() for n, t in self.tabs.items()}
        nav_htmls = []
        body_htmls = []

        for key, value in tabs.items():
            nav_htmls.append(f'<li onclick="handleLinkClick(this)">{html.escape(key)}</li>')
            # Can not escape here because this is HTML. Escaping it will present the HTML as text.
            # The renderer must ensure that the HTML is safe.
            body_htmls.append(f"<div>{value}</div>")

        template = string.Template(self.template_path.open("r").read())

        raw_html = template.substitute(NAV_HTML="".join(nav_htmls), BODY_HTML="".join(body_htmls))
        if ipython_check():
            try:
                from IPython.core.display import HTML

                return HTML(raw_html)
            except ImportError:
                ...
        return raw_html


def get_tab(name: str, /, create_if_missing: bool = True) -> Tab:
    """
    Get a tab by name. If the tab does not exist, create it.

    :param name: The name of the tab.
    :param create_if_missing: Whether to create the tab if it does not exist.
    :return: The tab.
    """
    report = current_report()
    return report.get_tab(name, create_if_missing=create_if_missing)


@syncify
async def log(content: str, do_flush: bool = False):
    """
    Log content to the main tab. The content should be a valid HTML string, but not a complete HTML document,
     as it will be inserted into a div.

    :param content: The content to log.
    :param do_flush: flush the report after logging.
    """
    get_tab(_MAIN_TAB_NAME).log(content)
    if do_flush:
        await flush.aio()


@syncify
async def flush():
    """
    Flush the report.
    """
    import flyte.storage as storage
    from flyte._context import internal_ctx
    from flyte._internal.runtime import io

    if not internal_ctx().is_task_context():
        return

    report = internal_ctx().get_report()
    if report is None:
        return

    report_html = report.get_final_report()
    assert report_html is not None
    assert isinstance(report_html, str)
    report_path = io.report_path(internal_ctx().data.task_context.output_path)
    content_types = {
        "Content-Type": "text/html",  # For s3
        "content_type": "text/html",  # For gcs
    }
    final_path = await storage.put_stream(report_html.encode("utf-8"), to_path=report_path, attributes=content_types)
    logger.debug(f"Report flushed to {final_path}")


@syncify
async def replace(content: str, do_flush: bool = False):
    """
    Get the report. Replaces the content of the main tab.

    :return: The report.
    """
    report = current_report()
    if report is None:
        return
    report.get_tab(_MAIN_TAB_NAME).replace(content)
    if do_flush:
        await flush.aio()


def current_report() -> Report:
    """
    Get the current report. This is a dummy report if not in a task context.

    :return: The current report.
    """
    from flyte._context import internal_ctx

    report = internal_ctx().get_report()
    if report is None:
        report = Report("dummy")
    return report
