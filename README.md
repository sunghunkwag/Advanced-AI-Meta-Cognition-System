# Advanced AI System: Neuro-Chemical Meta-Cognition

> **From Chaos to Order: The Birth of Intrinsic Will**

This repository houses an **autonomous AI agent** that learns by combining **Logical Reasoning** (System 2, causal inference & GAT) and **Neuro-Chemical Reinforcement Learning** (Dopamine/Serotonin drive). It is designed for experimenting with intrinsic motivation, logical axioms, and self-organized crystallization of knowledge.

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
python main_system_enhanced.py --config experimental --save-metrics results/exp.json
```

See [SETUP.md](SETUP.md) for detailed instructions and workflows.

---

## Core Pillars & Modules

| Pillar   | Goal                        | File(s)           | Mechanism                                                             |
|----------|-----------------------------|-------------------|-----------------------------------------------------------------------|
| Body     | Dexterity & Action          | action_decoder.py | Dual-Head NN: [Action Type, Raw Coordinates]                          |
| Mind     | Logical Reasoning           | manifold.py, vision.py | GAT, Graph-based Perception, Truth Vector (axiom injection)      |
| Heart    | Intrinsic Motivation        | energy_improved.py | Dual-hormone: Dopamine/Serotonin, Enhanced JEPA world model           |
| Soul     | Knowledge Crystallization   | automata.py, soul.py| Elastic Weight Consolidation (EWC), Nirvana condition                 |
| Config   | Experiment Management       | config.py         | Centralized parameters, device/seed/logging, curriculum               |
| Logging  | Metrics & Tracing           | logger.py         | Structured metrics, summary/statistics/tracking                       |
| Eval     | Analysis & Comparison       | evaluation.py      | Curve/statistics/action distribution, convergence, phase transitions  |

---

## Key Features
- **Intrinsic RL**: Driven by energy reduction, truth alignment, hormone homeostasis.
- **Logical Reasoning**: Causal, relational reasoning using graph attention and axioms.
- **JEPA World Model**: Predicts latent state/energy for future planning (model-based RL ready).
- **Configurable Workflow**: Centralized hyperparams, flexible curriculum, tracking, device select.
- **Curriculum & Boredom Penalty**: Encourages exploration early, prevents local minima.
- **Knowledge Crystallization**: If stable in truth+energy, locks in knowledge ("Nirvana").
- **Full Experiment Logging**: Structured, replayable metrics with deep analysis.

---

## Run Modes

- **Basic:** `python main_system.py` — original simple system
- **Enhanced:** `python main_system_enhanced.py` — config/logging/JEPA/analysis
- **Analysis:** `python evaluation.py results/exp.json` — print full report
- **Batch Compare:** `python evaluation.py results/*.json` — multi-run comparison.

---

## Example: Quick Enhanced Run

```bash
python main_system_enhanced.py --steps 200 --config experimental --device cpu --seed 42 --save-metrics results/test.json
python evaluation.py results/test.json
```

---

## Architecture Diagram

Perception → Mind (GAT + Soul) → Heart (Neuro-Chemical) → Soul (EWC) → Body (Action) → World
     ↑                                                                             ↓
     └─────────────────────── Feedback/Crystallization Loop ───────────────────────┘

---

## 📗 Documentation & API
- Modular config via `config.py` ([see example](SETUP.md)).
- Run `python main_system_enhanced.py --help` for all CLI options.
- All logs and metrics are JSON serializable; compatible with Python and analysis tools.
- For advanced config: Import `SystemConfig` and modify parameters before instantiating the agent.

---

## 🔬 Research & Citations
If you use this system, cite as:
```bibtex
@software{advanced_ai_metacognition,
    title = {Advanced AI Meta-Cognition System},
    author = {Kwag, Sunghun},
    year = {2025},
    url = {https://github.com/sunghunkwag/Advanced-AI-Meta-Cognition-System}
}
```

MIT Licensed. See LICENSE for details.

See [SETUP.md](SETUP.md) for full usage, troubleshooting, and configuration tips.
