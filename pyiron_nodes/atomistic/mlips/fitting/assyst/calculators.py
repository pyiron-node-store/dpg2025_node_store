from dataclasses import dataclass

from pyiron_workflow import Workflow

from pyiron_nodes.atomistic.mlips.calculator._generic import AseCalculatorConfig

from ase import Atoms

@Workflow.wrap.as_dataclass_node
@dataclass
class GenericOptimizerSettings:
    max_steps: int = 10
    force_tolerance: float = 1e-2

from ase.filters import StrainFilter, FrechetCellFilter
from ase.constraints import FixSymmetry, FixAtoms

import numpy as np
from enum import Enum

class RelaxMode(Enum):
    VOLUME = "volume"
    # CELL = "cell"
    # POSITION = "position"
    FULL = "full"

    def apply_filter_and_constraints(self, structure):
        match self:
            case RelaxMode.VOLUME:
                structure.set_constraint(FixAtoms(np.ones(len(structure),dtype=bool)))
                return FrechetCellFilter(structure, hydrostatic_strain=True)
            # case RelaxMode.CELL:
            #     return FrechetCellFilter(structure, constant_volume=True)
            case RelaxMode.FULL:
                return FrechetCellFilter(structure)
            case _:
                raise ValueError("Lazy Marvin")


@Workflow.wrap.as_function_node
def Relax(mode: str, calculator: AseCalculatorConfig, opt: GenericOptimizerSettings.dataclass, structure: Atoms) -> Atoms:
    from ase.optimize import LBFGS
    from ase.calculators.singlepoint import SinglePointCalculator

    mode = RelaxMode(mode)

    structure = structure.copy()

    # FIXME: meh
    match mode:
        case RelaxMode.VOLUME:
            structure.calc = calculator.get_calculator(use_symmetry=True)
        case RelaxMode.FULL:
            structure.calc = calculator.get_calculator(use_symmetry=False)
        case _:
            assert False

    filtered_structure = mode.apply_filter_and_constraints(structure)
    lbfgs = LBFGS(filtered_structure, logfile="/dev/null")
    lbfgs.run(fmax=opt.force_tolerance, steps=opt.max_steps)
    calc = structure.calc
    structure.calc = SinglePointCalculator(structure, **{
            'energy': calc.get_potential_energy(),
            'forces': calc.get_forces(),
            'stress': calc.get_stress()
    })
    # play catch with nodes
    relaxed_structure = structure
    relaxed_structure.constraints.clear()
    return relaxed_structure

# from pyiron_workflow.nodes.for_loop import For
# from pyiron_workflow.nodes.static_io import StaticNode
# from typing import ClassVar

# class RelaxLoop(For):
#     _body_node_class: ClassVar[type[StaticNode]] = Relax
#     _iter_on: ClassVar[tuple[str, ...]] = ("structure",)
#     _output_as_dataframe: ClassVar[bool] = False

@Workflow.wrap.as_function_node
def RelaxLoop(
        mode: str | RelaxMode,
        calculator: AseCalculatorConfig,
        opt: GenericOptimizerSettings.dataclass,
        structures: list[Atoms]
) -> list[Atoms]:
    from tqdm.auto import tqdm
    mode = RelaxMode(mode)
    relaxed_structures = []
    for structure in tqdm(structures, desc=f"Relax {mode.value}"):
        relaxed_structures.append(
                Relax.node_function(mode, calculator, opt, structure)
        )
    return relaxed_structures
