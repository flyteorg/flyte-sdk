import sys
from typing import Any


def _is_fastapi_app(app: Any) -> bool:
    """
    Return True if app is a FastAPI app.
    """
    try:
        fastapi = sys.modules["fastapi"]
    except KeyError:
        return False
    return isinstance(app, fastapi.FastAPI)
