from importlib.metadata import version

__version__ = version('madas')

# Commonly used classes
from .material import Material  # noqa: F401
from .data_framework import MaterialsDatabase  # noqa: F401
from .similarity import SimilarityMatrix  # noqa: F401
from .fingerprint import Fingerprint  # noqa: F401
