# Advanced AI Meta-Cognition System

Autonomous AI agent combining logical reasoning (Graph Attention Networks) with neuro-chemical reinforcement learning (dopamine/serotonin dynamics). Designed for research in intrinsic motivation, logical axioms, and knowledge crystallization.

## Quick Start

```bash
pip install -r requirements.txt
python main_system_fixed.py
```

For advanced usage with configuration management:

```bash
python main_system_enhanced.py --config experimental --steps 200 --save-metrics results/run.json
python evaluation.py results/run.json
```

## System Architecture

| Component | Purpose | Implementation | Key Features |
|-----------|---------|----------------|-------------|
| Body | Action execution | action_decoder.py | Dual-head network, REINFORCE sampling |
| Mind | Logical reasoning | manifold.py, vision.py | Graph Attention Network, truth vectors |
| Heart | Motivation | energy_improved.py | Dopamine/serotonin dynamics, JEPA predictor |
| Soul | Consolidation | automata.py, soul.py | Elastic Weight Consolidation (EWC) |
| World | Environment | world.py | 2D grid with symmetry-based energy |
| Config | Parameters | config.py | Centralized hyperparameter management |
| Logger | Metrics | logger.py | Structured experiment tracking |
| Eval | Analysis | evaluation.py | Statistical analysis and comparison |

## Core Features

**Learning Algorithm**: REINFORCE (policy gradient) for gradient flow through non-differentiable environment

**Intrinsic Motivation**: Dual-hormone system (dopamine for exploration, serotonin for stability)

**Logical Consistency**: Graph-based reasoning with injected truth axioms

**Knowledge Preservation**: EWC-based crystallization when system reaches stable low-energy states

**Curriculum Learning**: Bootstrapped exploration in early training phases

**Experiment Management**: Comprehensive configuration system with reproducible seeds

## Implementation Status

### Current (Fixed) Implementation

- `main_system_fixed.py` - Correct REINFORCE implementation with proper gradient flow
- `main_system_enhanced.py` - Full system with configuration and logging (REINFORCE applied)
- `action_decoder.py` - GPU-compatible action sampling with log probabilities
- `world.py` - Improved energy calculation with density control

### Legacy Files

- `main_system.py` - Original implementation (gradient flow issue, reference only)
- `main_asi.py` - Deprecated prototype (import errors)

### Critical Fixes Applied

**Gradient Flow**: Implemented REINFORCE algorithm to enable learning through non-differentiable environment. Previous implementation used `torch.tensor(world_energy)` which broke the computational graph.

**GPU Compatibility**: Added `.cpu()` calls before `.numpy()` conversions to prevent CUDA tensor errors.

**Action Sampling**: Added `sample_action()` method that returns both action and log probability for policy gradient calculation.

**Dimension Consistency**: Standardized feature dimensions (nfeat=3) across all components.

**Energy Calculation**: Improved density control with over-saturation penalties and localized CLEAR actions.

See [CRITICAL_FIXES.md](CRITICAL_FIXES.md) for detailed technical explanations.

## Usage Examples

### Basic Training Run

```bash
python main_system_fixed.py
```

Expected output: Energy reduction of 30-70% over 50 steps, grid density stabilizing around 10-20%.

### Configured Experiment

```bash
python main_system_enhanced.py --steps 1000 --seed 42 --device cuda --save-metrics exp1.json
```

### Analysis and Comparison

```bash
python evaluation.py exp1.json
python evaluation.py exp1.json exp2.json exp3.json  # Compare multiple runs
```

### Custom Configuration

```python
from config import SystemConfig
from main_system_enhanced import AdvancedAISystem

config = SystemConfig()
config.training.max_steps = 500
config.architecture.latent_dim = 16
config.neurochemical.energy_threshold = 5.0

system = AdvancedAISystem(config)
logger = system.run()
```

## Performance Characteristics

**Execution Speed**: 5000-7000 steps/sec on CPU, 10000+ on GPU

**Memory Usage**: Under 200MB for default configuration

**Convergence**: Typically 30-70% energy reduction within 100-500 steps

**Stability**: Tested up to 20,000 steps with 96%+ variance reduction

## Testing

Run comprehensive test suite:

```bash
python test_system.py
```

Tests cover:
- Component initialization and imports
- PyTorch setup and device detection
- Individual module functionality
- Gradient flow verification
- Full system integration (10-step run)

Expected: 8/8 tests passing

## File Structure

```
.
├── main_system_fixed.py          # Primary implementation (use this)
├── main_system_enhanced.py       # With config/logging (use this)
├── main_system.py                # Legacy reference
├── action_decoder.py             # Action network (fixed)
├── world.py                      # Environment (improved)
├── vision.py                     # Perception
├── manifold.py                   # GAT reasoning
├── energy_improved.py            # Neuro-chemical + JEPA
├── automata.py                   # EWC crystallization
├── soul.py                       # Truth vectors
├── config.py                     # Configuration system
├── logger.py                     # Metrics tracking
├── evaluation.py                 # Analysis tools
├── test_system.py                # Test suite
├── CRITICAL_FIXES.md             # Technical fix documentation
├── SETUP.md                      # Detailed setup guide
├── RUN_TEST.md                   # Testing instructions
└── requirements.txt              # Dependencies
```

## Technical Details

### REINFORCE Implementation

The system uses policy gradient (REINFORCE) to learn through a non-differentiable environment:

```python
action, log_prob = body.sample_action(action_logits, params)
world.apply_action(action)
reward = -world.calculate_energy()  # Negative energy as reward
policy_loss = -log_prob * reward
loss = policy_loss + consistency_loss + ewc_loss
loss.backward()  # Gradient flows correctly
```

### Energy Function

World energy combines symmetry error and density deviation:

```python
energy = symmetry_error + |density - 0.1| * 20 + over_saturation_penalty
```

Lower energy indicates more ordered, symmetric configurations at target density.

### Hormone Dynamics

Dopamine increases with energy reduction (exploration drive), serotonin increases with stability and low energy (exploitation phase).

## Configuration

Key hyperparameters in `config.py`:

- `latent_dim`: Latent space dimensionality (default: 8)
- `learning_rate`: Optimizer learning rate (default: 0.01)
- `energy_threshold`: Serotonin activation threshold (default: 0.2)
- `dopamine_boost_factor`: Dopamine sensitivity (default: 2.0)
- `curriculum_draw_steps`: Forced exploration steps (default: 10)

See [TUNING_GUIDE.md](TUNING_GUIDE.md) for recommended parameter ranges based on 20,000-step testing.

## Known Issues and Limitations

**Serotonin Collapse**: In long runs (>5000 steps), serotonin may drop to zero if thresholds are too strict. Adjust `energy_threshold` and `stability_variance_threshold` in config.

**Density Overshoot**: May exceed 10% target. Tune `boredom_penalty_weight` and over-saturation penalties.

**No Crystallization**: Requires achieving stable low-energy state. May need longer runs or adjusted thresholds.

See [TUNING_GUIDE.md](TUNING_GUIDE.md) for solutions.

## Citation

```bibtex
@software{advanced_ai_metacognition,
    title = {Advanced AI Meta-Cognition System},
    author = {Kwag, Sunghun},
    year = {2025},
    url = {https://github.com/sunghunkwag/Advanced-AI-Meta-Cognition-System}
}
```

## License

MIT License. See LICENSE file for details.

## Documentation

- [SETUP.md](SETUP.md) - Installation and configuration
- [CRITICAL_FIXES.md](CRITICAL_FIXES.md) - Technical implementation details
- [TUNING_GUIDE.md](TUNING_GUIDE.md) - Hyperparameter tuning based on long-term testing
- [RUN_TEST.md](RUN_TEST.md) - Testing procedures
- [TEST_RESULTS_20K.md](TEST_RESULTS_20K.md) - 20,000-step validation results
