import numpy as np

import landau

from pyiron_workflow import as_function_node


@as_function_node(use_cache=False)
def TransitionTemperature(
        phase1, phase2,
        Tmin: int | float,
        Tmax: int | float,
        dmu: int | float | None = 0,
        plot: bool = True
) -> float:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython import display
    df = landau.calculate.calc_phase_diagram([phase1, phase2], np.linspace(Tmin, Tmax), dmu, keep_unstable=True)
    try:
        fm, Tm = df.query('border and T!=@Tmin and T!=@Tmax')[['f','T']].iloc[0].tolist()
    except IndexError:
        display("Transition Point not found!")
        fm, Tm = np.nan, np.nan
    if plot:
        sns.lineplot(
            data=df,
            x='T', y='f',
            hue='phase',
            style='stable', style_order=[True, False],
        )
        plt.axvline(Tm, color='k', linestyle='dotted', alpha=.5)
        plt.scatter(Tm, fm, marker='o', c='k', zorder=10)

        dfa =  np.ptp(df['f'].dropna())
        dft =  np.ptp(df['T'].dropna())
        plt.text(Tm + .05 * dft, fm + dfa * .1, rf"$T_m = {Tm:.0f}\,\mathrm{{K}}$", rotation='vertical', ha='center')
        plt.xlabel("Temperature [K]")
        plt.ylabel("Free Energy [eV/atom]")
        plt.show()
    return Tm


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

    import landau
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
        prob = np.exp(-beta*(phis - conc*mu))
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


@as_function_node(use_cache=False)
def PlotConcPhaseDiagram(
        phase_data,
        plot_samples: bool = False,
        plot_isolines: bool = False,
        plot_tielines: bool = True,
        linephase_width: float = 0.01,
):
    """Plot a concentration-temperature phase diagram.

    phase_data should originate from CalcPhaseDiagram.

    Args:
        phases: list of phases to consider
        plot_samples (bool): overlay points where phase data has been sampled
        plot_isolines (bool): overlay lines of constance chemical potential
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import landau
    landau.plot.plot_phase_diagram(phase_data, min_c_width=0.01)
    if plot_samples:
        sns.scatterplot(
            data=phase_data,
            x='c', y='T',
            hue='phase',
            legend=False,
            s=1
        )
    if plot_isolines:
        sns.lineplot(
            data=phase_data.loc[np.isfinite(phase_data.mu)],
            x='c', y='T',
            hue='mu',
            units='phase', estimator=None,
            legend=False,
        )
    plt.xlabel("Temperature [K]")
    plt.show()


@as_function_node(use_cache=False)
def PlotMuPhaseDiagram(phase_data):
    """Plot a chemical potential-temperature phase diagram.

    phase_data should originate from CalcPhaseDiagram.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.scatterplot(
        data=phase_data.query('not border'),
        x='mu', y='T',
        hue='phase',
        s=5,
    )
    sns.scatterplot(
        data=phase_data.query('border'),
        x='mu', y='T',
        c='k',
        s=5,
    )
    plt.xlabel("Temperature [K]")
    plt.show()


@as_function_node(use_cache=False)
def PlotMuConcDiagram(phase_data):
    """Plot dependence of concentration on chemical potential in stable phases.

    phase_data should originate from CalcPhaseDiagram.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.lineplot(
        data=phase_data.query('stable'),
        x='mu', y='c',
        style='phase',
        hue='T',
    )
    plt.xlabel("Chemical Potential Difference [eV]")
    plt.show()
   

@as_function_node(use_cache=False)
def PlotPhiMuDiagram(phase_data):
    """Plot dependence of semigrand-potential on chemical potential in stable phases.

    phase_data should originate from CalcPhaseDiagram.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.lineplot(
        data=phase_data.query('stable'),
        x='mu', y='phi',
        style='phase',
        hue='T',
    )
    plt.xlabel("Semigrand Potential [eV/atom]")
    plt.xlabel("Chemical Potential Difference [eV]")
    plt.show()
