# Research Prototype Changelog

## Added
- Deterministic configuration module (`config.py`) and documented refactor plan for world-model-driven research.
- JEPA-like `WorldModel` with replay buffer, trainer, and deterministic simulation API for planner rollouts.
- System-2 planner with prefrontal arbitration hooks and System-3 `MetaLearner` adjusting optimizer, exploration, and EWC weighting.
- Experiment harness (`experiments/run_experiment.py`) that logs reproducible metrics to CSV.
- Unit and integration tests covering world-model determinism, planner ranking, and experiment log creation.

## Changed
- World environment now supports tensor state access and setting for simulated rollouts.
- Action decoder can encode actions for the world model and uses clearer parameter names.
- Neurochemical engine exposes dopamine gain for meta-learner tuning.
- Dependency list updated for experiment logging and testing.

## Verification
- `pytest` exercises planner ranking, world model determinism, and experiment logging.
- `experiments/run_experiment.py --steps 5` generates CSV logs demonstrating the integrated loop.
