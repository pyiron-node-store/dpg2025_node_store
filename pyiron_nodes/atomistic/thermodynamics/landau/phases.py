from pyiron_workflow import as_function_node
import landau

@as_function_node("phase")
def LinePhase(name: str, concentration: float, energy: float, entropy: float) -> landau.phases.LinePhase:
    import landau
    return landau.phases.LinePhase(name, concentration, energy, entropy)


@as_function_node("phase")
def IdealSolution(name: str, phase1: landau.phases.Phase, phase2: landau.phases.Phase) -> landau.phases.Phase:
    import landau
    return landau.phases.IdealSolution(name, phase1, phase2)
