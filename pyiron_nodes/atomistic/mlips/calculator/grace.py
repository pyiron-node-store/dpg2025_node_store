from dataclasses import dataclass
from functools import lru_cache

from pyiron_workflow import as_dataclass_node
from ._generic import AseCalculatorConfig

@as_dataclass_node
@dataclass(frozen=True, eq=True)
class Grace(AseCalculatorConfig):
    """Universal Graph Atomic Cluster Expansion models."""
    model: str = "GRACE-FS-OAM"

    @lru_cache(maxsize=1)
    def get_calculator(self, use_symmetry=True):
        # disable tensorflow warnings noise
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        from tensorpotential.calculator import grace_fm
        return grace_fm(self.model)
