# Refactor Plan: Advanced AI Meta-Cognition System

## Architecture Map (Current)
- `world.py`: 2D grid environment with drawing/symmetry actions and energy metric.
- `vision.py`: converts grid to graph nodes/adjacency for GAT-based mind.
- `manifold.py`: GraphAttentionManifold (System-1 reasoning) with truth/rejection vectors and consistency scoring.
- `action_decoder.py`: decodes latent state into action logits/parameters.
- `energy.py`: NeuroChemicalEngine (heart) updating dopamine/serotonin/cortisol from energy/consistency.
- `automata.py`: IntrinsicAutomata (soul) computing EWC loss and crystallization.
- `planner.py`: placeholder tree search not integrated with a predictive world model.
- `main_system.py`: procedural loop gluing components; limited determinism and no experiment harness.
- `test_multi_seed.py`: ad-hoc script for running multiple seeds; lacks assertions.

## Dead/Weak Points
- No deterministic seeding or experiment logging.
- Planner operates without a learned world model; System-2 arbitration absent.
- Meta-learning logic missing; parameters static after initialization.
- World model (JEPA) nonexistent; no rollout capability.
- Vision graph can explode in nodes; no clustering.
- Tests are minimal; no unit or integration coverage for new features.

## Refactor/Implementation Plan (Phases)
### Phase 1: World Model & Determinism
- Add `config.py` with seeds, planner/meta toggles, thresholds.
- Implement `world_model.py` (JEPA-like) that predicts next state/energy/consistency from `(state, action)` and supports deterministic `simulate` for planning.
- Add replay buffer collection during interaction; train interleaved.
- Extend `World` with `get_state_tensor()` and `set_state(state)` to support rollouts.
- Tests: world model forward determinism, shape checks, rollout without environment.

### Phase 2: Predictive Planning (System-2)
- Replace placeholder `planner.py` with model-based lookahead using the world model and clear objective (energy↓, consistency↑, cortisol penalty).
- Add `PrefrontalCortex` arbitration that triggers System-2 override based on configurable cortisol/consistency thresholds and failure streaks.
- Logging: intervention rate and planner objective comparisons.
- Tests: mocked world model ranking, integration with arbitration.

### Phase 3: Meta-Learner (System-3)
- Create `meta_learner.py` to adjust live hyperparameters (optimizer lr, exploration epsilon, EWC lambda, dopamine gain) using time-series features (energy trend, consistency volatility, action diversity, crystallization frequency).
- Apply updates on schedule; log parameter trajectories.
- Tests: meta updates modify parameters in-place and respect bounds.

### Phase 4: Vision Graph Scalability
- Add connected-component clustering to `vision.py` to collapse occupied cells into nodes with centroid/mass features and self-looped adjacency.
- Metric: average node count before/after, regression test for known patterns.

### Phase 5: Experiment Harness & Reporting
- Add `experiments/run_experiment.py` CLI with seeds, steps, planner/meta toggles, world model on/off, output dir.
- Logging to CSV/JSONL per run with energy, consistency, hormones, planner interventions, world-model losses.
- Add `make reproduce` (or documented command) for default experiment; include golden seed run with expected ranges.
- Tests: CLI smoke test (short horizon), metrics file creation.

### Phase 6: Documentation & Changelog
- Update README with methods, reproduction commands, expected metrics, deterministic setup, and environment definition.
- Add `docs/CHANGELOG_RESEARCH_PROTOTYPE.md` summarizing work and verification artifacts.

## Acceptance Criteria
- Deterministic experiments via single seed entry point.
- World model prediction error decreases over time; planner uses it for rollouts.
- System-2 interventions improve outcomes vs planner-off condition.
- Meta-learning alters hyperparameters and yields measurable effect across seeds.
- Reduced vision graph size with maintained performance.
- All new modules covered by unit/integration tests; default `pytest` passes.
