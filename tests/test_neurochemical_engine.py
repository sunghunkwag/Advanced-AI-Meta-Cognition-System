from energy import NeuroChemicalEngine


def test_neurochemical_engine_updates_and_clamps():
    engine = NeuroChemicalEngine()

    # Run several updates to exercise boredom, chaos, and recovery paths
    for _ in range(3):
        engine.update(world_energy=0.5, consistency_score=0.8, density=0.2, symmetry=0.6, prediction_error=0.0)
    for _ in range(3):
        engine.update(world_energy=0.5, consistency_score=0.2, density=0.8, symmetry=0.1, prediction_error=0.9)

    hormones = engine.get_hormones()

    # Values should remain within [0, 1]
    for value in hormones.values():
        assert 0.0 <= value <= 1.0

    # Ensure cortisol reacted to boredom/chaos signals
    assert hormones["cortisol"] > 0.0

    # Dopamine and serotonin should respond to provided signals rather than stay at defaults
    assert hormones["dopamine"] != 0.5 or hormones["serotonin"] != 0.5
