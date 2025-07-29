from __future__ import annotations

import pathlib
import sys
import typing

import pytest

from flyte._code_bundle._utils import list_imported_modules_as_files
from flyte._utils.lazy_module import is_imported, lazy_module

if typing.TYPE_CHECKING:
    import pandas as pd


# This test only works if the lazy_module declaration is inside the test body. seems like pytest is doing something
# that actually triggers loading.
@pytest.mark.skip("lazy module is not working in pytest")
def test_lazy_declaration():
    pd = lazy_module("pandas")

    def xx(df: pd.DataFrame):
        return pd.DataFrame(...)

    assert not is_imported("pandas")


@pytest.mark.skip("lazy module is not working in pytest")
def test_lazy_module_checking():
    lazy_module("pandas")
    assert not is_imported("pandas")

    sys_modules = list(sys.modules.values())
    res = list_imported_modules_as_files(str(pathlib.Path(__file__).parent.parent), sys_modules)
    assert not is_imported("pandas")
    print(res)
