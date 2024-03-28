import importlib.metadata

from evalverse.evaluator import Evaluator

__version__ = importlib.metadata.version("evalverse")

__all__ = [Evaluator]
