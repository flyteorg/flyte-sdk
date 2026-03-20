"""
Pandera data validation plugin for Flyte.

This plugin integrates Pandera's runtime data validation with Flyte's type system,
enabling automatic validation of pandas DataFrames against Pandera schemas when
data flows between tasks.

.. autosummary::
   :template: custom.rst
   :toctree: generated/

   PanderaTransformer
   PandasReportRenderer
   ValidationConfig
"""

from .config import ValidationConfig as ValidationConfig
from .renderer import PandasReportRenderer as PandasReportRenderer
from .transformer import PanderaTransformer as PanderaTransformer
