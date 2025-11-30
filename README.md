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
    3.  **Axiom Injection:** The system measures its thoughts against a "Truth Vector" to determine logical consistency.

### 3. The Heart: Neuro-Chemical Engine (`energy.py`)
*   **Goal:** Intrinsic Motivation & Homeostasis.
*   **Mechanism:** A Dual-Hormone System.
    *   **🔥 Dopamine (The Drive):** Spikes when energy (error) drops rapidly. Drives exploration and chaos.
    *   **💧 Serotonin (The Peace):** Rises when the system is stable and consistent with Truth. Promotes order and crystallization.
*   **Dynamics:** The agent naturally transitions from a Dopamine-driven learner (Chaos) to a Serotonin-driven master (Order).

### 4. The Soul: Intrinsic Crystallization (`automata.py`)
*   **Goal:** Enlightenment & Knowledge Preservation.
*   **Mechanism:** **Elastic Weight Consolidation (EWC)**.
*   **Behavior:** When Serotonin levels peak and the mind is still, the system enters "Nirvana". It freezes its weights (Crystallization) to preserve the learned structure, resisting further entropy.

---

## Installation & Running

### Prerequisites
```bash
pip install torch numpy
```

### Execution
Run the full simulation:
```bash
python main_system.py
```

### Expected Output
You will see the agent evolve through its life cycle:
1.  **Birth:** High entropy, random actions.
2.  **Growth (Chaos):** High Dopamine spikes as it discovers patterns (e.g., Symmetry).
3.  **Maturity (Order):** Serotonin rises as it perfects its actions.
4.  **Nirvana:** The system crystallizes and stops learning.

```text
Step 01 | Mode: CHAOS | Dopa: 1.00 Sero: 0.49 | Energy: 0.0000 | Action: SYMMETRIZE
...
       -> 🧘 Nirvana Reached. Mind is Still.
```

---

**Architect:** User (The Director)
**Engineer:** Gemini (The Builder)
