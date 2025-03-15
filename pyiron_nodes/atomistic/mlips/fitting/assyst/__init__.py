from pyiron_workflow import Workflow
from pyiron_workflow.nodes.standard import Multiply

from .structures import (
        SpaceGroupSampling,
        ElementInput,
        CombineStructures,
        SaveStructures
)
from .random import RattleLoop, StretchLoop
from pyiron_nodes.atomistic.relax import GenericOptimizerSettings, Relax, RelaxLoop
from pyiron_nodes.atomistic.structure.view import PlotSPG
from pyiron_nodes.atomistic.mlips.calculator.grace import Grace

def make_assyst(
        name, *elements,
        min_atoms=2, max_atoms=4, max_structures=50,
        delete_existing_savefiles=False
):
    wf = Workflow(name, delete_existing_savefiles=delete_existing_savefiles)
    if wf.has_saved_content():
        return wf

    element_nodes = []
    if len(elements) > 0:
        e1, *elements = elements
        stoi = ElementInput(e1, min_atoms=min_atoms, max_atoms=max_atoms)
        setattr(wf, 'Element_1', stoi)
        element_nodes.append(stoi)
        for i, e in enumerate(elements):
            en = ElementInput(e, min_atoms=min_atoms, max_atoms=max_atoms)
            setattr(wf, f'Element_{i+2}', en)
            element_nodes.append(en)
            stoi = Multiply(stoi, en)
            setattr(wf, f'Multiply_{i+1}_{i+2}', stoi)
        if len(elements) > 0:
            wf.Multiply = stoi
        spg = SpaceGroupSampling(
                elements=stoi,
                spacegroups=None,
                max_atoms=len(element_nodes) * max_atoms,
                max_structures=max_structures
        )
    else:
        spg = SpaceGroupSampling(
                max_atoms=max_atoms,
                max_structures=max_structures
        )
    plotspg = PlotSPG(spg)

    calc_config = Grace()
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

    wf.SpaceGroupSampling = spg
    wf.PlotSPG = plotspg

    wf.Calculator = calc_config
    wf.OptimizerSettings = optimizer_settings
    wf.VolumeRelax = volume_relax
    wf.FullRelax = full_relax

    wf.Rattle = rattle
    wf.Stretch = stretch

    wf.CombineStructures = combine_structures
    wf.SaveStructures = savestructures

    # wf.inputs_map = {
    #     'input__elements': 'elements',
    #     'input__max_atoms': 'max_atoms',
    #     'input__spacegroups': 'spacegroups',
    #     'input__min_dist': 'min_dist',
    #     'Calculator__model': None,
    #     'VolumeRelax__mode': None,
    #     'FullRelax__mode': None,
    #     'Stretch__samples': 'stretch_samples',
    # }
    # wf.outputs_map = {
    #     'sampling__structures': 'crystals',
    #     'volume_relax__relaxed_structure': 'volmin',
    #     'full_relax__relaxed_structure': 'allmin',
    #     'volume_relax__structure': None,
    #     'full_relax__structure': None,
    # }

    return wf
