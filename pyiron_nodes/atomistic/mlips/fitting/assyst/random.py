"""Nodes to randomly rattle and shake structures."""

from pyiron_workflow import Workflow

from ase import Atoms
import numpy as np


def rattle(structure: Atoms, sigma: float) -> Atoms:
    """Randomly displace positions with gaussian noise.

    Operates INPLACE."""
    structure.rattle(stdev=sigma)
    return structure


def stretch(structure: Atoms, hydro: float, shear: float) -> Atoms:
    """Randomly stretch cell with uniform noise.

    Operates INPLACE."""
    strain = shear * (2 * np.random.rand(3, 3) - 1)
    strain = 0.5 * (strain + strain.T)  # symmetrize
    np.fill_diagonal(strain, 1 + hydro * (2 * np.random.rand(3) - 1))
    structure.set_cell(structure.cell.array @ strain, scale_atoms=True)
    return structure


@Workflow.wrap.as_function_node
def Rattle(structure: Atoms, sigma: float, samples: int) -> list[Atoms]:
    structures = []
    # no point in rattling single atoms
    if len(structure) > 1:
        for _ in range(samples):
            structures.append(
                    stretch(
                        rattle(structure.copy(), sigma),
                        hydro=0.05, shear=0.005
                    )
            )
    return structures


@Workflow.wrap.as_function_node
def RattleLoop(
        structures: list[Atoms], sigma: float, samples: int
) -> list[Atoms]:
    rattled_structures = []
    for structure in structures:
        rattled_structures += Rattle.node_function(structure, sigma, samples)
    return rattled_structures


@Workflow.wrap.as_function_node
def Stretch(
        structure: Atoms, hydro: float, shear: float, samples: int,
        hydro_shear_ratio: float = 0.7
) -> list[Atoms]:
    structures = []
    for _ in range(samples):
        if np.random.rand() < hydro_shear_ratio:
            ihydro, ishear = hydro, 0.05
        else:
            ihydro, ishear = 0.05, shear
        structures.append(
                stretch(structure.copy(), hydro=ihydro, shear=ishear)
        )
    return structures


@Workflow.wrap.as_function_node
def StretchLoop(
        structures: list[Atoms], hydro: float, shear: float, samples: int,
        hydro_shear_ratio: float = 0.7
) -> list[Atoms]:
    stretched_structures = []
    for structure in structures:
        stretched_structures += Stretch.node_function(
                structure, hydro, shear, samples, hydro_shear_ratio
        )
    return stretched_structures
