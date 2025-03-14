from dataclasses import dataclass
from functools import lru_cache

from pyiron_workflow import as_dataclass_node
from ._generic import AseCalculatorConfig

@dataclass
class PawDftInput:
    encut: int | float | None = 320.
    kspacing: float | None = 0.5

    scf_energy_convergence: float = 1e-2

@lru_cache(maxsize=2)
def _get_gpaw(encut: float, kspacing: float, scf_energy_convergence: float, use_symmetry: bool):
    import gpaw
    return gpaw.GPAW(
        xc='PBE',
        kpts=(1,1,1),
        h=.25,
        nbands=-2,
        mode=gpaw.PW(encut, dedecut='estimate'),
        #FIXME deliberately high values for testing
        convergence={
            'energy': scf_energy_convergence,
            'density': 1,
            'eigenstates': 1e-3,
        },
        symmetry={'point_group': use_symmetry},
        txt=None,
    )

@as_dataclass_node
@dataclass
class Gpaw(AseCalculatorConfig, PawDftInput):
    def get_calculator(self, use_symmetry=True):
        return _get_gpaw(
                self.encut, self.kspacing, self.scf_energy_convergence,
                use_symmetry
        )
