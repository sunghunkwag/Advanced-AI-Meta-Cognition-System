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
    *   **JEPA Predictor:** Allows the system to "imagine" the future (System 2) by predicting the next state given an action.

### 4. The Soul: Intrinsic Crystallization (`automata.py`)
*   **Goal:** Enlightenment & Knowledge Preservation.
*   **Mechanism:** **Elastic Weight Consolidation (EWC)**.
*   **Behavior:** When Serotonin levels peak and the mind is still, the system enters "Nirvana". It freezes its weights (Crystallization) to preserve the learned structure, resisting further entropy.

---

## System 2: Meta-Cognition & Planning

The system now possesses a **Meta-Cognitive Controller** that arbitrates between two modes of thinking:

*   **System 1 (Intuition):** Fast, reflexive actions driven by the `ActionDecoder`. Used when entropy is low and the environment is stable.
*   **System 2 (Deliberation):** Slow, planned actions. Triggered by **High Entropy** (Confusion) or **Unstable Energy** (Panic).
    *   **Mechanism:** The `TreeSearchPlanner` uses the `LatentWorldModel` (JEPA) to simulate future outcomes and select the action that minimizes predicted energy.

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
You will see the agent evolve through its life cycle, occasionally triggering System 2 when confused:

```text
Step 05 | Mode: CHAOS | Dopa: 0.85 Sero: 0.50 | Energy: 0.1200 | Action: DRAW
🛑 [System 2 Triggered] High Uncertainty (Entropy 1.82 > 1.50)
[System 2] Deliberating (Depth 2)...
   -> 🧠 Plan: [1]->[0]
Step 06 | Mode: CHAOS | Dopa: 0.82 Sero: 0.50 | Energy: 0.1100 | Action: SYMMETRIZE
...
       -> 🧘 Nirvana Reached. Mind is Still.
```

---

**Architect:** User (The Director)
**Engineer:** Gemini (The Builder)
