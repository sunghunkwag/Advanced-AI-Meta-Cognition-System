# Advanced AI System: Neuro-Chemical Reinforcement Learning

> **"From Chaos to Order. The Birth of Intrinsic Will."**

This repository houses the **Advanced AI System**, a fully autonomous agent driven by **Logical Reasoning** and **Neuro-Chemical Reinforcement Learning**. Unlike traditional RL agents that chase external rewards, this system is driven by internal hormonal dynamics (Dopamine & Serotonin) to find truth and symmetry.

## Core Architecture: The 4 Pillars

The system is built upon four distinct but interconnected modules, mirroring biological cognition.

### 1. The Body: Dexterous Action Decoder (`action_decoder.py`)
*   **Goal:** Spatial Intelligence & Dexterity.
*   **Mechanism:** A Dual-Head Neural Network.
    *   **Head 1 (Logits):** Decides *what* to do (Draw, Symmetrize, Clear, Noise).
    *   **Head 2 (Params):** Decides *where* and *how* (x, y, scale, axis).
*   **Key Feature:** **No Hardcoded Templates.** The agent must learn to output raw continuous coordinates to interact with the world.

### 2. The Mind: Graph Attention Manifold (`manifold.py`)
*   **Goal:** Relational Reasoning & Logical Consistency.
*   **Mechanism:** **Graph Attention Network (GAT)**.
*   **Process:**
    1.  **Perception (`vision.py`):** Extracts objects as Nodes and spatial relationships as Edges.
    2.  **Reasoning:** The GAT infers causal relationships between nodes.
    3.  **Axiom Injection:** The system measures its thoughts against a "Truth Vector" (from `soul.py`) to determine logical consistency.

### 3. The Heart: Neuro-Chemical Engine (`energy.py`)
*   **Goal:** Intrinsic Motivation & Homeostasis.
*   **Mechanism:** A 3-Hormone System.
    *   **🔥 Dopamine (The Drive):** Spikes when energy (error) drops rapidly. Drives exploration and learning.
    *   **💧 Serotonin (The Peace):** Rises when the system achieves meaningful order. Promotes satisfaction and stability.
    *   **⚡ Cortisol (The Stress):** Increases from boredom or chaos. Forces action to escape discomfort.

### 4. The Soul: Intrinsic Crystallization (`automata.py`)
*   **Goal:** Enlightenment & Knowledge Preservation.
*   **Mechanism:** **Elastic Weight Consolidation (EWC)**.
*   **Behavior:** When Serotonin levels peak and the mind is still, the system enters "Nirvana". It freezes its weights (Crystallization) to preserve the learned structure.

---

## Installation & Running

### Prerequisites
```bash
pip install torch numpy
```

### Execution

**1. Basic Simulation (Single Life):**
```bash
python main_system_dual.py
```

**2. Autonomous Evolution (Recursive Self-Improvement):**
This runs an infinite loop where the agent lives, learns, dies, and is reborn. The **Meta-Learner** persists across lives, optimizing the learning strategy (`meta_brain.pth`).
```bash
python autonomous_evolution.py
```
*Run this in the background (`nohup python autonomous_evolution.py &`) to let the AI evolve over days/weeks.*

**3. Validation Tests:**
```bash
python test_meta_learning.py  # Verify Meta-Learner performance
python test_multi_seed.py     # Verify robustness
```

---

## New: System 3 - Recursive Meta-Learning

> **"Learning How to Learn."**

The system has evolved beyond simple adaptation. It now possesses **Meta-Cognition**, enabling it to optimize its own learning process in real-time.

### The Meta-Learner (`meta_cognition.py`)
A higher-order LSTM network that observes the agent's learning trajectory and dynamically tunes the brain's plasticity.

**Inputs (Generation-Invariant):**
- Consistency Score (Mental Stability)
- Symmetry Score (Visual Order)
- Density (Occupancy)
- Last Action & Reward
- Gradient Norm (Learning Stability)
- Energy Delta (Progress)

**Outputs (Dynamic Hyperparameters):**
1.  **Learning Rate Scale (0.1x - 10x):** Should I learn fast (plasticity) or slow (stability)?
2.  **Cortisol Sensitivity:** How much should pain (stress) drive my updates?
3.  **Entropy Regularization:** Should I explore (curiosity) or exploit (habit)?

### The Meta-Loop
While the base agent optimizes for consistency and energy, the **Meta-Learner optimizes for long-term emotional well-being**.
- It uses **REINFORCE** (Policy Gradient) to maximize the cumulative signal: `Dopamine + Serotonin - Cortisol`.
- This creates a feedback loop where the agent *learns to learn* in a way that leads to peace (Nirvana) faster.

---

**Architect:** User (The Director)  
**Engineer:** Gemini (The Builder)  
**Status:** ✅ Verified & Functional

## License

MIT License
