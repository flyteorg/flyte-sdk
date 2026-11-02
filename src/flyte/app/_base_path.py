"""The path an app is served under, when its URL is path-based.

A platform can serve apps under a path of a host it shares with other apps
(self-hosted onebox at https://platform.example.com/onebox/apps/<name>/). It
then tells the app that path in `_U_APP_BASE_PATH`, and requests reach the
app with the path intact.
"""

from __future__ import annotations

import os
import typing

APP_BASE_PATH_ENV_VAR = "_U_APP_BASE_PATH"

# Streamlit reads every option from STREAMLIT_<SECTION>_<OPTION>: this is
# --server.baseUrlPath. A --server.baseUrlPath on the command line still wins.
STREAMLIT_BASE_URL_PATH_ENV_VAR = "STREAMLIT_SERVER_BASE_URL_PATH"


def app_base_path(environ: typing.Mapping[str, str] = os.environ) -> str:
    """The app's base path without its trailing slash, or "" when it is served at its host's root."""
    return environ.get(APP_BASE_PATH_ENV_VAR, "").rstrip("/")


def apply_base_path_defaults(environ: typing.MutableMapping[str, str] = os.environ) -> None:
    """Point the app servers that read their base path from the environment at the app's.

    Without it, Streamlit serves at / and answers the app's own path with a
    redirect that loops against the platform's. A value the app sets itself is
    kept.
    """
    base = app_base_path(environ)
    if base:
        environ.setdefault(STREAMLIT_BASE_URL_PATH_ENV_VAR, base)
