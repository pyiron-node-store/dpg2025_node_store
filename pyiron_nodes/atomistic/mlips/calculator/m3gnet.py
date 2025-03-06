from dataclasses import dataclass
from functools import lru_cache

from pyiron_workflow import as_dataclass_node
from ._generic import AseCalculatorConfig

GPA2EVA3 = 0.006_241_509_074
@as_dataclass_node
@dataclass
class M3gnet(AseCalculatorConfig):
    model: str = "M3GNet-MP-2021.2.8-PES"

    @lru_cache(maxsize=1)
    def get_calculator(self, use_symmetry=True):
        from matgl.ext.ase import M3GNetCalculator
        from matgl import load_model, models

        return M3GNetCalculator(load_model(self.model), compute_stress=True, stress_weight=GPA2EVA3)
