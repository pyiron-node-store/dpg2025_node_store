from dataclasses import dataclass
from functools import lru_cache

from pyiron_workflow import as_dataclass_node
from ._generic import AseCalculatorConfig

@as_dataclass_node
@dataclass(frozen=True, eq=True)
class Grace(AseCalculatorConfig):
    """Universal Graph Atomic Cluster Expansion models."""
    model: str = "GRACE-1L-OAM_2Feb25"

    @lru_cache(maxsize=1)
    def get_calculator(self, use_symmetry=True):
        from tensorpotential.calculator import grace_fm
        return grace_fm(self.model)
