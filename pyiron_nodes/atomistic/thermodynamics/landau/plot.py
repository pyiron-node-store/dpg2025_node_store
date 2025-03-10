import numpy as np

import landau

from pyiron_workflow import as_function_node


@as_function_node(use_cache=False)
def TransitionTemperature(
        phase1, phase2,
        Tmin: int | float,
        Tmax: int | float,
) -> float:
    """Plot free energies of two phases and find their intersection, i.e. the transition temperature.

    Assumes that both phases are of the same concentration, otherwise the results will be off, as it takes the chemical
    potential difference to be zero.

    Args:
        phase1, phase2 (landau.phases.Phase): the two phases to plot
        Tmin (float): minimum temperature
        Tmax (float): maximum temperature

    Returns:
        float: transition temperature if found, else NaN
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    df = landau.calculate.calc_phase_diagram([phase1, phase2], np.linspace(Tmin, Tmax), mu=0.0, keep_unstable=True)
    try:
        fm, Tm = df.query('border and T!=@Tmin and T!=@Tmax')[['f','T']].iloc[0].tolist()
    except IndexError:
        print("Transition Point not found!")
        fm, Tm = np.nan, np.nan
    sns.lineplot(
        data=df,
        x='T', y='f',
        hue='phase',
        style='stable', style_order=[True, False],
    )
    plt.axvline(Tm, color='k', linestyle='dotted', alpha=.5)
    plt.scatter(Tm, fm, marker='o', c='k', zorder=10)

    dfa = np.ptp(df['f'].dropna())
    dft = np.ptp(df['T'].dropna())
    plt.text(Tm + .05 * dft, fm + dfa * .1, rf"$T_m = {Tm:.0f}\,\mathrm{{K}}$", rotation='vertical', ha='center')
    plt.xlabel("Temperature [K]")
    plt.ylabel("Free Energy [eV/atom]")
    plt.show()
    return Tm


@as_function_node(use_cache=False)
def PlotConcPhaseDiagram(
        phase_data,
        plot_samples: bool = False,
        plot_isolines: bool = False,
        plot_tielines: bool = True,
        linephase_width: float = 0.01,
        concavity: float | None = None,
):
    """Plot a concentration-temperature phase diagram.

    phase_data should originate from CalcPhaseDiagram.

    Args:
        phases: list of phases to consider
        plot_samples (bool): overlay points where phase data has been sampled
        plot_isolines (bool): overlay lines of constance chemical potential
        plot_tielines (bool): add grey lines connecting triple points
        linephase_width (float): phases that have a solubility less than this
            will be plotted as a rectangle
        concavity (float, optional, range in [0, 1]): how aggressive to be when
            fitting polyhedra to samples phase data; lower means more ragged
            shapes, higher means smoother; 1 corresponds to convex hull of points
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import landau
    landau.plot.plot_phase_diagram(
            phase_data.drop('refined', errors='ignore', axis='columns'),
            min_c_width=linephase_width,
            alpha=concavity or 0.1,
    )
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
            sort=False,
        )
    if plot_tielines and 'refined' in phase_data.columns:
        # hasn't made it upstream yet
        for T, dd in phase_data.query('refined=="delaunay-triple"').groupby('T'):
            plt.plot(dd.c, [T]*3, c='k', alpha=.5, zorder=-10)
    plt.xlabel("Concentration")
    plt.ylabel("Temperature [K]")
    plt.show()


@as_function_node(use_cache=False)
def PlotMuPhaseDiagram(phase_data):
    """Plot a chemical potential-temperature phase diagram.

    phase_data should originate from CalcPhaseDiagram.
    Phase boundaries are plotted in black.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    border = None
    if 'border' not in phase_data.columns:
        body = phase_data.query('not border')
    else:
        border = phase_data.query('border')
        body = phase_data.query('not border')
    sns.scatterplot(
        data=body,
        x='mu', y='T',
        hue='phase',
        s=5,
    )
    if border is not None:
        sns.scatterplot(
            data=border,
            x='mu', y='T',
            c='k',
            s=5,
        )
    plt.xlabel("Chemical Potential Difference [eV]")
    plt.ylabel("Temperature [K]")
    plt.show()


@as_function_node(use_cache=False)
def PlotIsotherms(phase_data):
    """Plot concentration isotherms in stable phases.

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
    plt.xlabel("Chemical Potential Difference [eV]")
    plt.ylabel("Semigrand Potential [eV/atom]")
    plt.show()


@as_function_node(use_cache=False)
def CheckTemperatureInterpolation(
        phase: landau.phases.TemperatureDependentLinePhase,
        Tmin: float | None = None, Tmax: float | None = None,
):
    """Plot the free energy interpolation against data points.

    Args:
        phase (landau.phases.TemperatureDependentLinePhase): phase to check
        Tmin, Tmax (floats): temperature window to check
    """
    import numpy as np
    import matplotlib.pyplot as plt
    if Tmin is None:
        Tmin = np.min(phase.temperatures) * 0.9
    if Tmax is None:
        Tmax = np.max(phase.temperatures) * 1.1
    Ts = np.linspace(Tmin, Tmax, 50)
    l, = plt.plot(Ts, phase.line_free_energy(Ts), label="interpolation")
    # try to plot about 50 points
    n = max(int(len(phase.temperatures) // 50), 1)
    plt.scatter(
            phase.temperatures[::n],
            phase.free_energies[::n],
            c=l.get_color(),
            label="data",
    )
