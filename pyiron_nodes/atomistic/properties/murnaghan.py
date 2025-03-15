from abc import ABC, abstractmethod
from ase import Atoms
from ase.calculators.calculator import Calculator
from dataclasses import dataclass
from functools import lru_cache
from pyiron_workflow import as_function_node, as_dataclass_node

from pyiron_workflow import Workflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.mlips.calculator._generic import AseCalculatorConfig
from pyiron_nodes.atomistic.mlips.calculator.ace import Ace
from pyiron_nodes.atomistic.mlips.calculator.grace import Grace
from pyiron_nodes.atomistic.relax import Relax, GenericOptimizerSettings


@as_function_node
def CalculateEVCurve(
    structure: Atoms,
    calculator: AseCalculatorConfig,
    num_of_points: int = 7,
    vol_range: float = 0.3,
    per_atom: bool = True,
    opt: GenericOptimizerSettings | None = None,
):
    """
    Computes an energy vs volume (EV) curve for a given structure.

    Args:
        structure (Atoms): atomic structure
        calculator (AseCalculatorConfig): energy/force engine to use
        num_of_points (int): volume samples
        vol_range (float): minimum/maximum volumetric strain
        per_atom (float): output per atom quantities rather than supercell quantities

    Returns:
        DataFrame: columns 'volume', 'energy', 'ase_atoms'
    """
    from ase.optimize import BFGS
    from ase.constraints import ExpCellFilter

    volume_factors = np.linspace((1 - vol_range)**(1/3), (1.0 + vol_range)**(1/3), num_of_points)

    structure = structure.copy()

    if per_atom:
        nd = len(structure)
    else:
        nd = 1

    structure.calc = calculator.get_calculator(use_symmetry=False)
    initial_volume = structure.get_volume()

    data = {"volume": [], "energy": [], "ase_atoms":[]}

    for factor in volume_factors:
        scaled_structure = structure.copy()
        scaled_structure.set_cell(structure.cell * factor, scale_atoms=True)
        scaled_structure.calc = calculator.get_calculator(use_symmetry=False)

        if opt is not None:
            opt = BFGS(scaled_structure)
            # Relax atomic positions
            opt.run(fmax=opt.force_tolerance, steps=opt.max_steps)

        energy = scaled_structure.get_potential_energy()
        volume = scaled_structure.get_volume()

        data["volume"].append(volume/nd)
        data["energy"].append(energy/nd)
        data["ase_atoms"].append(scaled_structure)

    df = pd.DataFrame(data)
    return df


def birch_murnaghan(vol, E0, V0, B0, BP):
    """
    Birch-Murnaghan EOS.
    """
    E = E0 + (9.0*V0*B0)/16.0 * ( ((V0/vol)**(2.0/3.0)-1.0)**3.0 *BP +
        ((V0/vol)**(2.0/3.0)-1.0)**2.0 * (6.0-4.0*(V0/vol)**(2.0/3.0)))
    return E



@as_function_node
def FitBirchMurnaghanEOS(ev_curve_df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Fits the energy vs volume data to the Birch-Murnaghan EOS
    and returns a tuple of equilibrium properties: (E0, V0, B0).
    """
    from scipy.optimize import curve_fit

    volumes = ev_curve_df["volume"].values
    energies = ev_curve_df["energy"].values

    # Initial guesses for the fitting parameters
    V0_guess = volumes[np.argmin(energies)]
    E0_guess = min(energies)
    B0_guess = 1.0  # in eV/Å³
    B1_guess = 4.0

    popt, _ = curve_fit(birch_murnaghan, volumes, energies, p0=[E0_guess, V0_guess, B0_guess, B1_guess])
    E0, V0, B0, B1 = popt
    B0_GPa = B0 * 160.21766208  # Conversion factor

    return E0, V0, B0_GPa

def make_murnaghan_workflow(
    workflow_name: str,
    element_str: str,         # New parameter: element string to build the structure
    potential_path: str,
    delete_existing_savefiles=False,
    fmax: float = 0.01,
    max_steps: int = 500,
    num_of_points: int = 7,
    vol_range: float = 0.1,
    per_atom: bool = True,
    optimize: bool = False,
    xlabel: str = "Volume (Å³)",
    ylabel: str = "Energy (eV)",
    title: str = "Energy vs Volume Curve",
    fontsize: int = 12
):
    wf = Workflow(workflow_name, delete_existing_savefiles=delete_existing_savefiles)
    if wf.has_saved_content():
        return wf

    # 1. Build the structure using Bulk.
    # The Bulk node is expected to take an element string (e.g. "Ca") and output a structure.
    from pyiron_nodes.atomistic.structure.build import Bulk

    if element_str=="CaMg":
        structure = generate_CaMg_ortho()
    else:
        wf.build_structure = Bulk(element_str)
        structure = wf.build_structure.outputs.structure
    # The output of wf.build_structure is assumed to be available under 'structure'
    # (e.g. bulk_struc = wf.build_structure.pull() returns the structure).

    # 2. Instantiate the appropriate calculator node based on potential_path.
    if "GRACE" in potential_path.upper():
        wf.Calculator = Grace(model=potential_path)
    else:
        wf.Calculator = Ace(potential_file=potential_path)

    # 3. Optimize the structure.
    wf.optimize_settings = GenericOptimizerSettings(max_steps, fmax)
    wf.optimize_structure = Relax(
            mode="full",
            calculator=wf.Calculator,
            opt=wf.optimize_settings,
            structure=structure,  # use structure from Bulk node
    )

    # 4. Calculate the energy–volume (EV) curve on the optimized structure.
    wf.calculate_ev_curve = CalculateEVCurve(
        structure=wf.optimize_structure.outputs.relaxed_structure,
        calculator=wf.Calculator,
        num_of_points=num_of_points,
        vol_range=vol_range,
        per_atom=per_atom,
        opt=wf.optimize_settings if optimize else None
    )

    # 5. Fit the Birch–Murnaghan EOS to extract equilibrium properties.
    wf.fit_eos = FitBirchMurnaghanEOS(
        ev_curve_df=wf.calculate_ev_curve.outputs.df
    )

    # 6. Plot the computed EV curve.
    wf.plot_ev_curve = PlotEVCurve(
        ev_curve_df=wf.calculate_ev_curve.outputs.df,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        fontsize=fontsize
    )

    # Input mapping: Now the workflow-level inputs are numerical parameters and the element string.
    # (Assuming Bulk expects an input key "element_str"; adjust if necessary.)
    wf.inputs_map = {
        'build_structure__element_str': 'element_str',
        'optimize_settings__fmax': 'fmax',
        'optimize_settings__max_steps': 'max_steps',
        'calculate_ev_curve__num_of_points': 'num_of_points',
        'calculate_ev_curve__vol_range': 'vol_range',
        'calculate_ev_curve__per_atom': 'per_atom',
        'calculate_ev_curve__optimize': 'optimize',
        'plot_ev_curve__xlabel': 'xlabel',
        'plot_ev_curve__ylabel': 'ylabel',
        'plot_ev_curve__title': 'title',
        'plot_ev_curve__fontsize': 'fontsize'
    }

    # Outputs mapping: Here, we assume FitBirchMurnaghanEOS returns a tuple (E0, V0, B0)
    wf.outputs_map = {
        'fit_eos__0': 'E0',  # first element of the tuple
        'fit_eos__1': 'V0',  # second element
        'fit_eos__2': 'B0',  # third element
        'plot_ev_curve__fig': 'fig',
        'plot_ev_curve__ax': 'ax'
    }

    return wf


def generate_CaMg_ortho():
    from ase import Atoms
    # Lattice vectors
    cell = np.array([
        [3.692641, 0.0, 0.0000000000000002],
        [-0.0000000000000004, 5.809565, 0.0000000000000004],
        [0.0, 0.0, 5.9892969999999996]
    ])

    # Element symbols
    symbols = ['Ca', 'Ca', 'Mg', 'Mg']

    # Fractional coordinates
    scaled_positions = np.array([
        [0.0000000000000000, 0.0000000000000000, 0.0932210000000000],
        [0.5000000000000000, 0.5000000000000000, 0.9067790000000000],
        [0.5000000000000000, 0.0000000000000000, 0.5849690000000000],
        [0.0000000000000000, 0.5000000000000000, 0.4150310000000000]
    ])

    # Create ASE Atoms object
    atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)

    return atoms


@as_function_node(use_cache=False)
def PlotEVCurve(
    ev_curve_df: pd.DataFrame,
    xlabel: str = "Volume (Å³)",
    ylabel: str = "Energy (eV)",
    title: str = "Energy vs Volume Curve",
    fontsize: int = 12
):
    """
    Plots the Energy vs. Volume (EV) curve from a computed EV dataset.

    Args:
        ev_curve_df (pd.DataFrame): DataFrame containing 'volume' and 'energy' columns.
        xlabel (str, optional): Label for the x-axis. Defaults to "Volume (Å³)".
        ylabel (str, optional): Label for the y-axis. Defaults to "Energy (eV)".
        title (str, optional): Title of the plot. Defaults to "Energy vs Volume Curve".
        fontsize (int, optional): Font size for labels and title. Defaults to 12.

    Returns:
        fig, ax: The matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots()
    ax.plot(ev_curve_df['volume'], ev_curve_df['energy'], marker='o', linestyle='-', color='b')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    # ax.legend()
    plt.show()
