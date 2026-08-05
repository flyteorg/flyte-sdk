def ipython_check() -> bool:
    """
    Check if interface is launching from iPython (not colab)
    :return is_ipython (bool): True or False
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
    :return: True if running in IPython with ipywidgets support, False otherwise.
    """
    try:
        import ipywidgets  # noqa: F401

        return True
    except (ImportError, NameError):
        return False
