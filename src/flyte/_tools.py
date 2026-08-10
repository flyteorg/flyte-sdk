def ipython_check() -> bool:
    """
    Check if interface is launching from iPython (not colab)

    Returns:
        bool: True if an IPython shell is active, else False. Note that Colab runs an
            IPython kernel, so it also returns True despite the summary above.
    """
    import sys

    if "IPython" not in sys.modules:
        return False
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except (ImportError, NameError):
        return False


def ipywidgets_check() -> bool:
    """
    Check if the interface is running in IPython with ipywidgets support.

    Returns:
        True if running in IPython with ipywidgets support, False otherwise.
    """
    try:
        import ipywidgets  # noqa: F401

        return True
    except (ImportError, NameError):
        return False
