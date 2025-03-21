from dataclasses import dataclass, asdict, field
from typing import Optional, Union, Tuple, List
import numpy as np
import os
import random
import string
from pyiron_workflow import as_function_node, as_macro_node, as_dataclass_node
from ase import Atoms
import pandas as pd

@as_function_node("potentials")
def ListPotentials(structure):
    from pyiron_atomistics.lammps.potential import list_potentials as lp

    potentials = lp(structure)
    return potentials

@as_dataclass_node
@dataclass
class MD:
    """
    Molecular dynamics parameters.

    Attributes:
    -----------
    timestep: float
        https://calphy.org/en/latest/inputfile.html#timestep
    n_small_steps: int
        https://calphy.org/en/latest/inputfile.html#n-small-steps
    n_every_steps: int
        https://calphy.org/en/latest/inputfile.html#n-every-steps
    n_repeat_steps: int
        https://calphy.org/en/latest/inputfile.html#n-repeat-steps
    n_cycles: int
        https://calphy.org/en/latest/inputfile.html#n-cycles
    thermostat_damping: float
        https://calphy.org/en/latest/inputfile.html#thermostat-damping
    barostat_damping: float
        https://calphy.org/en/latest/inputfile.html#barostat-damping
    """
    timestep: float = 0.001
    n_small_steps: int = 10000
    n_every_steps: int = 10
    n_repeat_steps: int = 10
    n_cycles: int = 100
    thermostat_damping: float = 0.5
    barostat_damping: float = 0.1

@as_dataclass_node
@dataclass
class Tolerance:
    """
    Tolerance parameters.

    Attributes:
    -----------
    spring_constant: float
        https://calphy.org/en/latest/inputfile.html#tol-spring-constant
    solid_fraction: float
        https://calphy.org/en/latest/inputfile.html#tol-solid-fraction
    liquid_fraction: float
        https://calphy.org/en/latest/inputfile.html#tol-liquid-fraction
    pressure: float
        https://calphy.org/en/latest/inputfile.html#tol-pressure
    """
    spring_constant: float = 0.01
    solid_fraction: float = 0.7
    liquid_fraction: float = 0.05
    pressure: float = 1.0

@as_dataclass_node
@dataclass
class NoseHoover:
    """
    Nose-Hoover parameters.

    Attributes:
    -----------
    thermostat_damping: float
        https://calphy.org/en/latest/inputfile.html#nose-hoover-thermostat-damping
    barostat_damping: float
        https://calphy.org/en/latest/inputfile.html#nose-hoover-barostat-damping
    """
    thermostat_damping: float = 0.1
    barostat_damping: float = 0.1

@as_dataclass_node
@dataclass
class Berendsen:
    """
    Berendsen parameters.

    Attributes:
    -----------
    thermostat_damping: float
        https://calphy.org/en/latest/inputfile.html#berendsen-thermostat-damping
    barostat_damping: float
        https://calphy.org/en/latest/inputfile.html#berendsen-barostat-damping
    """
    thermostat_damping: float = 100.0
    barostat_damping: float = 100.0

@as_dataclass_node
@dataclass
class InputClass:
    """
    Input parameters for calphy calculations.

    Attributes:
    -----------
    md: MD
        Molecular dynamics parameters.
    tolerance: Tolerance
        Tolerance parameters.
    nose_hoover: NoseHoover
        Nose-Hoover parameters.
    berendsen: Berendsen
        Berendsen parameters.
    queue: Queue
        Queue parameters.
    pressure: int
        https://calphy.org/en/latest/inputfile.html#pressure
    temperature: int
        https://calphy.org/en/latest/inputfile.html#temperature
    npt: bool
        https://calphy.org/en/latest/inputfile.html#npt
    n_equilibration_steps: int
        https://calphy.org/en/latest/inputfile.html#n-equilibration-steps
    n_switching_steps: int
        https://calphy.org/en/latest/inputfile.html#n-switching-steps
    n_print_steps: int
        https://calphy.org/en/latest/inputfile.html#n-print-steps
    n_iterations: int
        https://calphy.org/en/latest/inputfile.html#n-iterations
    equilibration_control: str
        https://calphy.org/en/latest/inputfile.html#equilibration-control
    melting_cycle: bool
        https://calphy.org/en/latest/inputfile.html#melting-cycle
    spring_constants: Optional[float]
        https://calphy.org/en/latest/inputfile.html#spring-constants        
    """
    md: Optional[MD] = None 
    tolerance: Optional[Tolerance] = None
    nose_hoover: Optional[NoseHoover] = None
    berendsen: Optional[Berendsen] = None
    pressure: int = 0
    temperature: int = 300
    temperature_stop: int = 600
    npt: bool = True
    n_equilibration_steps: int = 2500
    n_switching_steps: int = 2500
    n_print_steps: int = 1000
    n_iterations: int = 1
    equilibration_control: str = "nose-hoover"
    melting_cycle: bool = False
    cores: Optional[int] = 1

def _generate_random_string(length: str) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def _prepare_potential_and_structure(potential, structure):
    from pyiron_atomistics.lammps.potential import LammpsPotential, LammpsPotentialFile
    from pyiron_atomistics.lammps.structure import (
        LammpsStructure,
    ) 

    potential_df = LammpsPotentialFile().find_by_name(potential)
    potential = LammpsPotential()
    potential.df = potential_df

    pair_style = []
    pair_coeff = []
    
    pair_style.append(" ".join(potential.df["Config"].to_list()[0][0].strip().split()[1:]))
    pair_coeff.append(" ".join(potential.df["Config"].to_list()[0][1].strip().split()[1:]))

    #now prepare the list of elements
    elements = potential.get_element_lst()
    elements_from_pot = potential.get_element_lst()

    lmp_structure = LammpsStructure()
    lmp_structure.potential = potential
    lmp_structure.atom_type = "atomic"
    lmp_structure.el_eam_lst = potential.get_element_lst()
    lmp_structure.structure = structure

    elements_object_lst = structure.get_species_objects()
    elements_struct_lst = structure.get_species_symbols()

    masses = []
    for element_name in elements_from_pot:
        if element_name in elements_struct_lst:
            index = list(elements_struct_lst).index(element_name)
            masses.append(elements_object_lst[index].AtomicMass)
        else:
            masses.append(1.0)

    file_name = os.path.join(os.getcwd(), _generate_random_string(7)+'.dat')
    lmp_structure.write_file(file_name=file_name)
    potential.copy_pot_files(os.getcwd())
    return pair_style, pair_coeff, elements, masses, file_name

def _prepare_input(inp, potential, structure, mode='fe', reference_phase='solid'):
    from calphy.input import Calculation
    import os
    pair_style, pair_coeff, elements, masses, file_name = _prepare_potential_and_structure(potential, structure)

    inpdict = asdict(inp)
    inpdict["pair_style"] = pair_style
    inpdict["pair_coeff"] = pair_coeff
    inpdict["element"] = elements
    inpdict["mass"] = masses
    inpdict['mode'] = mode
    inpdict['reference_phase'] = reference_phase
    inpdict['lattice'] = file_name
    inpdict["queue"] = {"cores": inpdict["cores"],}
    del inpdict["cores"]

    if inpdict["md"] is None:
        inpdict["md"] = {
                "timestep": 0.001,
                "n_small_steps": 10000,
                "n_every_steps": 10,
                "n_repeat_steps": 10,
                "n_cycles": 100,
                "thermostat_damping": 0.5,
                "barostat_damping": 0.1,
        }
    if inpdict["tolerance"] is None:
        inpdict["tolerance"] = {
                "spring_constant": 0.01,
                "solid_fraction": 0.7,
                "liquid_fraction": 0.05,
                "pressure": 1.0,
        }
    if inpdict["nose_hoover"] is None:
        inpdict["nose_hoover"] = {
                "thermostat_damping": 0.1,
                "barostat_damping": 0.1,
        }
    if inpdict["berendsen"] is None:
        inpdict["berendsen"] = {
                "thermostat_damping": 100.0,
                "barostat_damping": 100.0,
        }
    if mode == 'ts':
        inpdict["temperature"] = [inpdict['temperature'], inpdict["temperature_stop"]]
        del inpdict["temperature_stop"]
        
    calc = Calculation(**inpdict)
    return calc

def _run_cleanup(simfolder, lattice, delete_folder=False):
    import shutil
    import os
    os.remove(lattice)
    if delete_folder:
        shutil.rmtree(simfolder)

@as_function_node('free_energy')
def SolidFreeEnergy(inp, structure: Atoms, potential: str) -> float:
    """
    Calculate the free energy of a solid phase.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.
    
    Returns:
    --------
    float
        Free energy in eV/atom
    """
    from calphy.solid import Solid
    from calphy.routines import routine_fe
    import os

    calc = _prepare_input(inp, potential, structure, mode='fe', reference_phase='solid')
    #os.chdir()
    simfolder = calc.create_folders()
    job = Solid(calculation=calc, simfolder=simfolder)
    job = routine_fe(job)
    _run_cleanup(simfolder, calc.lattice)
    return job.report["results"]["free_energy"]

@as_function_node('free_energy')
def LiquidFreeEnergy(inp, structure: Atoms, potential: str) -> float:
    """
    Calculate the free energy of a liquid phase.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.
    
    Returns:
    --------
    float
        Free energy in eV/atom
    """
    from calphy.liquid import Liquid
    from calphy.routines import routine_fe
    
    calc = _prepare_input(inp, potential, structure, mode='fe', reference_phase='liquid')
    simfolder = calc.create_folders()
    job = Liquid(calculation=calc, simfolder=simfolder)
    job = routine_fe(job)
    #run calculation
    _run_cleanup(simfolder, calc.lattice)
    return job.report["results"]["free_energy"]

@as_function_node('temperature', 'free_energy')
def SolidFreeEnergyWithTemp(inp, structure: Atoms, potential: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the free energy of a solid phase as a function of temperature.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Temperature and free energy in K and eV/atom, respectively.
    """
    from calphy.solid import Solid
    from calphy.routines import routine_ts
    
    calc = _prepare_input(inp, potential, structure, mode='ts', reference_phase='solid')
    simfolder = calc.create_folders()
    job = Solid(calculation=calc, simfolder=simfolder)
    job = routine_ts(job)
    #run calculation

    #grab the results
    datafile = os.path.join(os.getcwd(), simfolder, 'temperature_sweep.dat')
    t, f = np.loadtxt(datafile, unpack=True, usecols=(0,1))

    _run_cleanup(simfolder, calc.lattice)
    return t, f

@as_function_node('temperature', 'free_energy')
def LiquidFreeEnergyWithTemp(inp, structure: Atoms, potential: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the free energy of a liquid phase as a function of temperature.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Temperature and free energy in K and eV/atom, respectively.
    """
    from calphy.liquid import Liquid
    from calphy.routines import routine_ts
    
    calc = _prepare_input(inp, potential, structure, mode='ts', reference_phase='liquid')
    simfolder = calc.create_folders()
    job = Liquid(calculation=calc, simfolder=simfolder)
    job = routine_ts(job)
    
    #grab the results
    datafile = os.path.join(os.getcwd(), simfolder, 'temperature_sweep.dat')
    t, f = np.loadtxt(datafile, unpack=True, usecols=(0,1))

    _run_cleanup(simfolder, calc.lattice)
    return t, f

@as_function_node
def PlotFreeEnergy(temperature: np.ndarray, free_energy: np.ndarray):
    import matplotlib.pyplot as plt
    plt.plot(temperature, free_energy, label='free energy')
    plt.ylabel('Free energy (eV/atom)')
    plt.xlabel('Temperature (K)')
    plt.legend(frameon=False)
    figure = plt.show()
    return figure

@as_function_node('phase_transition_temperature', 'figure')
def CalcPhaseTransformationTemp(temp_A: np.ndarray, fe_A: np.ndarray, temp_B: np.ndarray, fe_B: np.ndarray, fit_order: int = 4):
    """
    Calculate the phase transformation temperature from free energy data.

    Parameters:
    -----------
    temp_A: np.ndarray
        Temperature array for phase 1.
    fe_A: np.ndarray
        Free energy array for phase 1.
    temp_B: np.ndarray
        Temperature array for phase 2.
    fe_B: np.ndarray
        Free energy array for phase 2.
    fit_order: int
        Order of the polynomial fit.
    
    Returns:
    --------
    float
        Phase transformation temperature
    """
    import matplotlib.pyplot as plt
    import warnings

    #do some fitting to determine temps
    t1min = np.min(temp_A)
    t2min = np.min(temp_B)
    t1max = np.max(temp_A)
    t2max = np.max(temp_B)

    tmin = np.min([t1min, t2min])
    tmax = np.max([t1max, t2max])

    #warn about extrapolation
    if not t1min == t2min:
        warnings.warn(f'free energy is being extrapolated!')
    if not t1max == t2max:
        warnings.warn(f'free energy is being extrapolated!')

    #now fit
    f1fit = np.polyfit(temp_A, fe_A, fit_order)
    f2fit = np.polyfit(temp_B, fe_B, fit_order)

    #reevaluate over the new range
    fit_t = np.arange(tmin, tmax+1, 1)
    fit_f1 = np.polyval(f1fit, fit_t)
    fit_f2 = np.polyval(f2fit, fit_t)

    #now evaluate the intersection temp
    arg = np.argsort(np.abs(fit_f1-fit_f2))[0]
    transition_temp = fit_t[arg]

    #warn if the temperature is shady
    if np.abs(transition_temp-tmin) < 1E-3:
        warnings.warn('It is likely there is no intersection of free energies')
    elif np.abs(transition_temp-tmax) < 1E-3:
        warnings.warn('It is likely there is no intersection of free energies')

    #plot
    c1lo = '#ef9a9a'
    c1hi = '#b71c1c'
    c2lo = '#90caf9'
    c2hi = '#0d47a1'

    plt.plot(fit_t, fit_f1, color=c1lo, label=f'phase A fit')
    plt.plot(fit_t, fit_f2, color=c2lo, label=f'phase B fit')
    plt.plot(temp_A, fe_A, color=c1hi, label='phase A', ls='dashed')
    plt.plot(temp_B, fe_B, color=c2hi, label='phase B', ls='dashed')
    plt.axvline(transition_temp, ls='dashed', c='#37474f')
    plt.ylabel('Free energy (eV/atom)')
    plt.xlabel('Temperature (K)')
    plt.legend(frameon=False)
    figure = plt.show()

    return transition_temp, figure

@as_function_node('results')
def CollectResults() -> pd.DataFrame:
    from calphy.postprocessing import gather_results
    df = gather_results('.')
    return df
