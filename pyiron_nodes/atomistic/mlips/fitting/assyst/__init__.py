from pyiron_workflow import Workflow
from pyiron_workflow.nodes.standard import Multiply

from .structures import (
        SpaceGroupInput,
        SpaceGroupSampling,
        ElementInput,
        StoichiometryTable,
        CombineStructures,
        SaveStructures
)
from .random import RattleLoop, StretchLoop
from .calculators import M3gnetConfig, GenericOptimizerSettings, Relax, RelaxLoop
from .plot import PlotSPG

def make_assyst(name, *elements, delete_existing_savefiles=False):
    # TODO!
    max_ion = 4
    max_structures = 200

    wf = Workflow(name, delete_existing_savefiles=delete_existing_savefiles)
    if wf.has_saved_content():
        return wf

    element_nodes = []
    if len(elements) > 0:
        e1, *elements = elements
        stoi = ElementInput(e1, max_ion=max_ion)
        setattr(wf, 'element_1', stoi)
        element_nodes.append(stoi)
        for i, e in enumerate(elements):
            en = ElementInput(e, max_ion=max_ion)
            setattr(wf, f'element_{i+2}', en)
            element_nodes.append(en)
            stoi = Multiply(stoi, en)
        if len(elements) > 0:
            wf.stoichiometry = stoi

        inp = SpaceGroupInput(stoichiometry=stoi, max_structures=max_structures)
    else:
        inp = SpaceGroupInput()
    spg = SpaceGroupSampling(inp)
    plotspg = PlotSPG(spg)

    calc_config = M3gnetConfig()
    optimizer_settings = GenericOptimizerSettings()

    volume_relax = RelaxLoop(mode="volume", calculator=calc_config,
                             opt=optimizer_settings, structures=spg)
    full_relax = RelaxLoop(mode="full", calculator=calc_config,
                           opt=optimizer_settings,
                           structures=volume_relax.outputs.relaxed_structures)

    rattle = RattleLoop(
            structures=full_relax.outputs.relaxed_structures,
            sigma=0.25,
            samples=4
    )

    stretch = StretchLoop(
            structures=full_relax.outputs.relaxed_structures,
            hydro=0.8,
            shear=0.2,
            samples=4
    )

    combine_structures = CombineStructures(
            spg,
            volume_relax.outputs.relaxed_structures,
            full_relax.outputs.relaxed_structures,
            rattle,
            stretch
    )

    savestructures = SaveStructures(combine_structures, "data/Structures_Everything")

    wf.input = inp
    wf.sampling = spg
    wf.plotspg = plotspg

    wf.calc_config = calc_config
    wf.optimizer_settings = optimizer_settings
    wf.volume_relax = volume_relax
    wf.full_relax = full_relax

    wf.rattle = rattle
    wf.stretch = stretch

    wf.combine_structures = combine_structures
    wf.savestructures = savestructures

    wf.inputs_map = {
        'input__elements': 'elements',
        'input__max_atoms': 'max_atoms',
        'input__spacegroups': 'spacegroups',
        'input__min_dist': 'min_dist',
        'calc_config__model': None,
        'volume_relax__mode': None,
        'full_relax__mode': None,
    }
    # wf.outputs_map = {
    #     'sampling__structures': 'crystals',
    #     'volume_relax__relaxed_structure': 'volmin',
    #     'full_relax__relaxed_structure': 'allmin',
    #     'volume_relax__structure': None,
    #     'full_relax__structure': None,
    # }

    return wf
