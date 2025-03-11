from pyiron_workflow import as_function_node
import landau
import numpy as np


def guess_mu_range(phases, Tmax, samples):
    """Guess chemical potential window from the ideal solution.

    Searches numerically for chemical potentials which stabilize
    concentrations close to 0 and 1 and then use the concentrations
    encountered along the way to numerically invert the c(mu) mapping.
    Using an even c grid with mu(c) then yields a decent sampling of mu
    space so that the final phase diagram is described everywhere equally.

    Args:
        phases: list of phases to consider
        Tmax: temperature at which to estimate 
        samples: how many mu samples to return

    Returns:
        array of chemical potentials that likely cover the whole concentration space
    """

    import scipy.optimize as so
    import scipy.interpolate as si
    import numpy as np
    # semigrand canonical "average" concentration
    # use this to avoid discontinuities and be phase agnostic
    def c(mu):
        phis = np.array([p.semigrand_potential(Tmax, mu) for p in phases])
        conc = np.array([p.concentration(Tmax, mu) for p in phases])
        phis -= phis.min()
        beta = 1/(Tmax*8.6e-5)
        prob = np.exp(-beta*phis)
        prob /= prob.sum()
        return (prob * conc).sum()
    cc, mm = [], []
    mu0, mu1 = 0, 0
    while (ci := c(mu0)) > 0.001:
        cc.append(ci)
        mm.append(mu0)
        mu0 -= 0.05
    while (ci := c(mu1)) < 0.999:
        cc.append(ci)
        mm.append(mu1)
        mu1 += 0.05
    cc = np.array(cc)
    mm = np.array(mm)
    I = cc.argsort()
    cc = cc[I]
    mm = mm[I]
    return si.interp1d(cc, mm)(np.linspace(min(cc), max(cc), samples))


# Move to separate
@as_function_node('phase_data')
def CalcPhaseDiagram(
        phases: list,
        temperatures: list[float] | np.ndarray,
        chemical_potentials: list[float] | np.ndarray | int = 100,
        refine: bool = True
):
    """Calculate thermodynamic potentials and respective stable phases in a range of temperatures.

    The chemical potential range is chosen automatically to cover the full concentration space.

    Args:
        phases: list of phases to consider
        temperatures: temperature samples
        mu_samples: number of samples in chemical potential space
        refine (bool): add additional sampling points along exact phase transitions

    Returns:
        dataframe with phase data
    """
    import matplotlib.pyplot as plt
    import landau

    if isinstance(chemical_potentials, int):
        mus = guess_mu_range(phases, max(temperatures), chemical_potentials)
    else:
        mus = chemical_potentials
    df = landau.calculate.calc_phase_diagram(
            phases, np.asarray(temperatures), mus,
            refine=refine, keep_unstable=False
    )
    return df
