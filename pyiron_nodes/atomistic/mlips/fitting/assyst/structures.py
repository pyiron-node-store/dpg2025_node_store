from dataclasses import dataclass
from collections.abc import Generator, Sequence
from numbers import Integral
from itertools import product

from pyiron_workflow import Workflow, as_function_node

import pandas as pd
from ase import Atoms

@dataclass(frozen=True)
class Elements(Sequence):
    atoms: tuple[dict[str, int]]

    @property
    def elements(self) -> set[str]:
        """Set of elements present in elements."""
        e = set()
        for s in self.atoms:
            e = e.union(s.keys())
        return e

    # FIXME: Self only availabe in >=3.11
    def __add__(self, other: 'Elements') -> 'Elements':
        """Extend underlying list of stoichiometries."""
        return Elements(self.atoms + other.atoms)

    def __or__(self, other: 'Elements') -> 'Elements':
        """Inner product of underlying stoichiometries.

        Must not share elements with other.elements."""
        assert self.elements.isdisjoint(other.elements), "Can only or stoichiometries of different elements!"
        s = ()
        for me, you in zip(self.atoms, other.atoms):
            s += (me | you,)
        return Elements(s)

    def __mul__(self, other: 'Elements') -> 'Elements':
        """Outer product of underlying stoichiometries.

        Must not share elements with other.elements."""
        assert self.elements.isdisjoint(other.elements), "Can only multiply stoichiometries of different elements!"
        s = ()
        for me, you in product(self.atoms, other.atoms):
            s += (me | you,)
        return Elements(s)

    # Sequence Impl'
    def __getitem__(self, index: int) -> dict[str, int]:
        return self.atoms[index]

    def __len__(self) -> int:
        return len(self.atoms)


@as_function_node
def ElementInput(
        element: str,
        min_atoms: int =  1,
        max_atoms: int = 10,
        step_atoms: int = 1,
) -> Elements:
    """Create a Elements object.

    Can be combined with other Elements to create a sampling in chemical space
    for :class`.SpaceGroupSampling`.
    Each entry gives the number of atoms of an element

    Args:
        element (str): canonical symbol for element
        min_atoms, max_atoms, step_atoms (int): range of number of atoms

    Returns:
        Elements: kj
    """
    elements = Elements(tuple({element: i} for i in range(min_atoms, max_atoms + 1, step_atoms)))
    return elements

@as_function_node("filtered")
def FilterSize(
        elements: Elements,
        min: int | None = 0,
        max: int | None = None,
):
    """Filter an Elements object by size.

    Args:
        min (int): new object has at least this number of atoms
        max (int): new object has at most this number of atoms

    Returns:
        Elements: filtered object
    """
    import math
    if max is None:
        max = math.inf
    return Elements(tuple(s for s in elements
                                if min <= sum(s.values()) <= max ))

@as_function_node(use_cache=False)
def ElementsTable(elements: Elements) -> pd.DataFrame:
    """Display the number of atoms inside an `Elements` object as a table."""
    import pandas as pd
    from IPython.display import display
    df = pd.DataFrame(elements.atoms)
    with pd.option_context("display.max_rows",    10000000,  # inf not allowed
                           "display.max_columns", 10000000): # just pick large
        display(df)

@as_function_node
def SpaceGroupSampling(
        elements: Elements,
        spacegroups: list[int] | tuple[int,...] | None = None,
        max_atoms: int = 10,
        max_structures: int | None = None,
) -> list[Atoms]:
    """
    Create symmetric random structures.

    Args:
        elements (Elements): list of compositions per structure
        spacegroups (list of int): which space groups to generate
        max_atoms (int): do not generate structures larger than this
        max_structures (int): generate at most this many structures
    Returns:
        list of Atoms: generated structures
    """
    from warnings import catch_warnings
    from structuretoolkit.build.random import pyxtal
    from tqdm.auto import tqdm
    import math

    if spacegroups is None:
        spacegroups = list(range(1,231))
    if max_structures is None:
        max_structures = math.inf

    structures = []
    with catch_warnings(category=UserWarning, action='ignore'):
        for stoich in (bar := tqdm(elements)):
            elements, num_atomss = zip(*stoich.items())
            stoich_str = "".join(f"{s}{n}" for s, n in zip(elements, num_atomss))
            bar.set_description(stoich_str)
            structures += [s['atoms'] for s in pyxtal(spacegroups, elements, num_atomss)]
            if len(structures) > max_structures:
                structures = structures[:max_structures]
                break
    return structures


@as_function_node
def CombineStructures(
        spacegroups: list[Atoms] | None,
        volume_relax: list[Atoms] | None,
        full_relax: list[Atoms] | None,
        rattle: list[Atoms] | None,
        stretch: list[Atoms] | None,
) -> list[Atoms]:
    """Combine individual structure sets into a full training set."""
    from functools import reduce
    structures = [spacegroups, volume_relax, full_relax, rattle, stretch]
    structures = reduce(list.__add__, (s or [] for s in structures), [])
    if len(structures) == 0:
        logging.warn("Either no inputs given or all inputs are empty. "
                     "Returning the empty list!")
    return structures


@as_function_node
def SaveStructures(
        structures: list[Atoms],
        filename: str
):
    """Save list of structures into a pickled dataframe.

    Columns are:
        'name': a structure label
        'ase_atoms': the ASE object for the actual structure
        'number_of_atoms': the number of atoms inside the structure

    If `filename` does not end with 'pckl.gz', it is added.

    Args:
        structures (list of Atoms): structures to save
        filename (str): path where the dataframe is written to
    """
    import pandas as pd
    import os.path
    df = pd.DataFrame([
        {'name': s.info.get('label', f'structure_{i}'),
         'ase_atoms': s,
         'number_of_atoms': len(s),
         } for i, s in enumerate(structures)])
    if not filename.endswith("pckl.gz"):
        filename += ".pckl.gz"
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    df.to_pickle(filename)
