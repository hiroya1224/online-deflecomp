# src/online_deflecomp/__init__.py
from importlib.metadata import version as _version, PackageNotFoundError

try:
    # 注意: 配布名はハイフン（プロジェクト名）＝ "online-deflecomp"
    __version__ = _version("online-deflecomp")
except PackageNotFoundError:
    # 未インストールのソース直読みなどに備えてフォールバック
    __version__ = "0.0.0"

from .pipeline import run_estimation_pipeline

__all__ = ["run_estimation_pipeline", "__version__"]
