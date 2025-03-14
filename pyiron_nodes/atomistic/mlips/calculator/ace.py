from dataclasses import dataclass
from functools import lru_cache

from pyiron_workflow import as_dataclass_node
from ._generic import AseCalculatorConfig

@as_dataclass_node
@dataclass(frozen=True, eq=True)
class Ace(AseCalculatorConfig):
    """Atomic Cluster Expansion Models."""

    potential_file: str

    @lru_cache(maxsize=1)
    def get_calculator(self, use_symmetry=True):
        from pyace import PyACECalculator
        calc = PyACECalculator(self.potential_file)
        from logging import ERROR
        from pyiron_snippets.logger import logger
        logger.setLevel(ERROR)
        return calc
