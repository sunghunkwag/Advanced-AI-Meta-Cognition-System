# Advanced AI Meta-Cognition System - Setup Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/sunghunkwag/Advanced-AI-Meta-Cognition-System.git
cd Advanced-AI-Meta-Cognition-System

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Run with default configuration
python main_system_enhanced.py

# Run with experimental configuration (larger network, more steps)
python main_system_enhanced.py --config experimental

# Run with custom parameters
python main_system_enhanced.py --steps 100 --seed 42 --device cuda

# Save metrics for later analysis
python main_system_enhanced.py --save-metrics results/run1.json
```

### 3. Legacy System

The original system is still available:

```bash
python main_system.py
```

## Configuration System

The enhanced system uses `config.py` for centralized configuration management.

### Configuration Presets

**Default Configuration:**
- Latent dimension: 8
- Max steps: 200
- Learning rate: 0.01
- Suitable for quick experiments

**Experimental Configuration:**
- Latent dimension: 16
- Max steps: 500
- Enhanced neuro-chemical sensitivity
- Better for observing complex emergent behaviors

### Custom Configuration

Create your own configuration:

```python
from config import SystemConfig, ArchitectureConfig, TrainingConfig

config = SystemConfig()
config.architecture.latent_dim = 32
config.training.max_steps = 1000
config.training.learning_rate = 0.005

from main_system_enhanced import AdvancedAISystem
system = AdvancedAISystem(config)
logger = system.run()
```

## Analysis and Evaluation

### Analyzing a Single Run

```bash
# Generate comprehensive analysis report
python evaluation.py results/run1.json
```

Output includes:
- Summary statistics
- Convergence analysis
- Phase transition detection
- Hormone correlation
- Action distribution
- Stability metrics

### Comparing Multiple Runs

```bash
# Compare different configurations
python evaluation.py results/run1.json results/run2.json results/run3.json
```

### Programmatic Analysis

```python
from evaluation import MetricsAnalyzer

analyzer = MetricsAnalyzer('results/run1.json')

# Get convergence step
conv_step = analyzer.get_convergence_step(threshold=0.3)

# Analyze phase transitions
transitions = analyzer.get_phase_transitions()

# Get action distribution
actions = analyzer.get_action_distribution()

# Generate full report
report = analyzer.generate_report()
print(report)
```

## Architecture Overview

### Core Components

1. **World** (`world.py`) - 2D grid environment with symmetry-based energy
2. **Vision** (`vision.py`) - Converts grid state to graph representation
3. **Mind** (`manifold.py`) - Graph Attention Network for reasoning
4. **Heart** (`energy_improved.py`) - Neuro-chemical engine (Dopamine/Serotonin)
5. **Body** (`action_decoder.py`) - Action selection network
6. **Soul** (`automata.py`) - Knowledge crystallization via EWC

### New Components

7. **Config** (`config.py`) - Centralized configuration management
8. **Logger** (`logger.py`) - Structured metrics tracking
9. **Evaluation** (`evaluation.py`) - Analysis utilities
10. **JEPA** (`energy_improved.py`) - World model predictor

## File Structure

```
.
├── main_system.py              # Original system
├── main_system_enhanced.py     # Enhanced system with config
├── config.py                   # Configuration management
├── logger.py                   # Metrics and logging
├── evaluation.py               # Analysis tools
├── energy_improved.py          # Enhanced neuro-chemical + JEPA
├── world.py                    # Environment
├── vision.py                   # Perception
├── manifold.py                 # Graph reasoning
├── action_decoder.py           # Action network
├── automata.py                 # Crystallization
├── soul.py                     # Truth vectors
├── requirements.txt            # Dependencies
├── README.md                   # Main documentation
└── SETUP.md                    # This file
```

## Experimental Workflow

### 1. Run Baseline

```bash
python main_system_enhanced.py --save-metrics results/baseline.json
```

### 2. Run Variations

```bash
# Longer training
python main_system_enhanced.py --steps 500 --save-metrics results/long.json

# Different seed
python main_system_enhanced.py --seed 123 --save-metrics results/seed123.json

# Experimental config
python main_system_enhanced.py --config experimental --save-metrics results/exp.json
```

### 3. Compare Results

```bash
python evaluation.py results/*.json
```

### 4. Analyze Best Run

```bash
python evaluation.py results/best_run.json
```

## Advanced Features

### Gradient Clipping

The enhanced system includes gradient clipping to prevent instability:

```python
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

### Layer Normalization

Improved JEPA uses LayerNorm for more stable training:

```python
nn.LayerNorm(hidden_dim)
```

### Curriculum Learning

Bootstraps learning with forced DRAW actions:

- Steps 1-10: 100% DRAW actions
- Steps 11-20: 50% DRAW actions
- Steps 21+: Free exploration

### World Model (JEPA)

The enhanced system includes a world model predictor that can:
- Predict next latent states
- Forecast energy costs
- Enable model-based planning (future work)

## Troubleshooting

### Issue: NaN losses

**Solution:** Reduce learning rate or enable gradient clipping

```bash
python main_system_enhanced.py --config default  # Uses 0.01 LR with clipping
```

### Issue: No crystallization

**Solution:** Run longer or adjust serotonin threshold

```python
config = get_default_config()
config.crystallization.serotonin_threshold = 0.9  # Lower threshold
config.training.max_steps = 1000  # More time
```

### Issue: Empty world (low grid activity)

**Solution:** Adjust curriculum or boredom penalty

```python
config.training.curriculum_draw_steps = 20  # More bootstrapping
config.world.boredom_penalty_weight = 1.0   # Stronger penalty
```

## Citation

If you use this system in your research, please cite:

```bibtex
@software{advanced_ai_metacognition,
  title = {Advanced AI Meta-Cognition System},
  author = {Kwag, Sunghun},
  year = {2025},
  url = {https://github.com/sunghunkwag/Advanced-AI-Meta-Cognition-System}
}
```

## License

MIT License - See LICENSE file for details
