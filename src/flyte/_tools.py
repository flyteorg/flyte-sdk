def ipython_check() -> bool:
    """
    Check if interface is launching from iPython (not colab)
    :return is_ipython (bool): True or False
    """
    is_ipython = False
    try:  # Check if running interactively using ipython.
        from IPython import get_ipython

        if get_ipython() is not None:
            is_ipython = True
    except (ImportError, NameError):
        pass
    return is_ipython


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
