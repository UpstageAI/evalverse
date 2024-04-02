import importlib.metadata

from evalverse.evaluator import Evaluator
from evalverse.reporter import Reporter

__version__ = importlib.metadata.version("evalverse")

__all__ = [Evaluator, Reporter]
