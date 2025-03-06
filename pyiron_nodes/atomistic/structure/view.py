from pyiron_workflow import as_function_node

from ase import Atoms as _Atoms
import numpy as _np

from typing import Iterable, Optional


@as_function_node("plot")
def Plot3d(
    structure: _Atoms,
    camera: str = "orthographic",
    particle_size: Optional[int | float] = 1.0,
    select_atoms: Optional[_np.ndarray] = None,
    view_plane: _np.ndarray = _np.array([0, 0, 1]),
    distance_from_camera: Optional[int | float] = 1.0,
):
    """Display atomistic structure (ase._Atoms) using nglview"""
    from structuretoolkit import plot3d

    return structure.plot3d(
        # structure=structure,
        camera=camera,
        particle_size=particle_size,
        select_atoms=select_atoms,
        view_plane=view_plane,
        distance_from_camera=distance_from_camera,
    )


@as_function_node(use_cache=False)
def PlotSPG(structures: list[_Atoms]):
    """Plot a histogram of space groups in input list."""
    from structuretoolkit.analyse import get_symmetry
    import matplotlib.pyplot as plt
    spacegroups = []
    for structure in structures:
        spacegroups.append(get_symmetry(structure).info['number'])
    plt.hist(spacegroups)
    plt.show()


@as_function_node(use_cache=False)
def PlotAtomsHistogram(structures: list[_Atoms]):
    """
    Plot a histogram of the number of atoms in each structure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    length = np.array([len(s) for s in structures])
    lo = length.min()
    hi = length.max()
    # make the bins fall in between whole numbers and include hi
    plt.hist(length, bins=np.arange(lo, hi + 2) - 0.5)
    plt.xlabel("#Atoms")
    plt.ylabel("Count")
    plt.show()


@as_function_node(use_cache=False)
def PlotAtomsCells(structures: list[_Atoms], angle_in_degrees: bool = True):
    """
    Plot histograms of cell parameters.

    Plotted are atomic volume, density, cell vector lengths and cell vector angles in separate subplots all on a
    log-scale.

    Args:
        structures (list of Atoms): structures to plot
        angle_in_degrees (bool): whether unit for angles is degree or radians

    Returns:
        `DataFrame`: contains the plotted information in the columns:
                        - a: length of first vector
                        - b: length of second vector
                        - c: length of third vector
                        - alpha: angle between first and second vector
                        - beta: angle between second and third vector
                        - gamma: angle between third and first vector
                        - V: volume of the cell
                        - N: number of atoms in the cell
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    N = np.array([len(s) for s in structures])
    C = np.array([s.cell.array for s in structures])

    # def get_angle(cell, idx=0):
    get_angle = lambda cell, idx=0: np.arccos(
            np.dot(cell[idx], cell[(idx + 1) % 3])
            / np.linalg.norm(cell[idx])
            / np.linalg.norm(cell[(idx + 1) % 3])
        )

    # def extract(n, c):
    extract = lambda n, c: {
            "a": np.linalg.norm(c[0]),
            "b": np.linalg.norm(c[1]),
            "c": np.linalg.norm(c[2]),
            "alpha": get_angle(c, 0),
            "beta": get_angle(c, 1),
            "gamma": get_angle(c, 2),
        }

    df = pd.DataFrame([extract(n, c) for n, c in zip(N, C)])
    df["V"] = np.linalg.det(C)
    df["N"] = N
    if angle_in_degrees:
        df["alpha"] = np.rad2deg(df["alpha"])
        df["beta"] = np.rad2deg(df["beta"])
        df["gamma"] = np.rad2deg(df["gamma"])

    plt.subplot(1, 4, 1)
    plt.title("Volume")
    plt.hist(df.V / df.N, bins=20, log=True)
    plt.xlabel(r"$V$ [$\mathrm{\AA}^3$/atom]")

    plt.subplot(1, 4, 2)
    plt.title("Density")
    plt.hist(df.N / df.V, bins=20, log=True)
    plt.xlabel(r"$\rho$ [atom/$\mathrm{\AA^3}$]")

    plt.subplot(1, 4, 3)
    plt.title("Lengths")
    plt.hist([df.a, df.b, df.c], log=True)
    plt.xlabel(r"$a,b,c$ [$\mathrm{\AA}$]")

    plt.subplot(1, 4, 4)
    plt.title("Angles")
    plt.hist([df.alpha, df.beta, df.gamma], log=True)
    if angle_in_degrees:
        label = r"$\alpha,\beta,\gamma$ [$^\circ$]"
    else:
        label = r"$\alpha,\beta,\gamma$ [rad]"
    plt.xlabel(label)
    plt.tight_layout()
    plt.show()


@as_function_node(use_cache=False)
def PlotDistances(structures: list[_Atoms], bins: int | Iterable[float] = 50, num_neighbors: int = 50, normalize: bool = True):
    """Plot radial distribution of a list of structures.

    Args:
        structures (list of Atoms): structures to plot
        bins (int or iterable of floats): if int number of bins; if iterable of floats bin edges
    """
    from structuretoolkit import get_neighbors
    import matplotlib.pyplot as plt
    import numpy as np
    distances = []
    for structure in structures:
        distances.append(get_neighbors(structure, num_neighbors=num_neighbors).distances.ravel())
    distances = np.concatenate(distances)

    if normalize:
        plt.hist(
            distances,
            bins=bins,
            density=True,
            weights=1 / (4 * np.pi * distances**2),
        )
        plt.ylabel(r"Neighbor density [$\mathrm{\AA}^{-3}$]")
    else:
        plt.hist(distances, bins=bins)
        plt.ylabel("Neighbor count")
    plt.xlabel(r"Distance [$\mathrm{\AA}$]")
    plt.show()


@as_function_node(use_cache=False)
def PlotConcentration(structures: list[_Atoms]):
    """
    Plot histograms of the concentrations in each structure.

    Args:
        structures (list of Atoms): structures to take concentrations of
    """
    from collections import Counter
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    symbols = [Counter(s.symbols) for s in structures]
    elements = sorted(set.union(*(set(s) for s in symbols)))

    df = pd.DataFrame([{e: c[e]/sum(c.values()) for e in elements} for c in symbols])

    sns.histplot(
        data=df.melt(var_name="element", value_name="concentration"),
        x="concentration",
        hue="element",
        multiple="dodge",
    )

    plt.show()
