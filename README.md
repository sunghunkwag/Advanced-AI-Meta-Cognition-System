# Advanced AI Meta Cognition System (V4.0)

> **"From Reaction to Deliberation. The Inner Eye Awakens."**

**Advanced AI Meta Cognition System** (formerly Project Daedalus) represents the next leap in Artificial Super Intelligence (ASI) architecture. It evolves the agent from **System 1 (Reactive)** to **System 2 (Deliberate)** thinking.

## Core Philosophy: The Inner Eye

Unlike V3.5, which acted based on immediate intuition, **V4.0** possesses the ability to **simulate the future** in its mind before acting in reality.

### 🧠 System 1: Intuition (Fast)
- **Mechanism:** `ActionDecoder` (Policy Network)
- **Usage:** Routine tasks, low uncertainty.
- **Speed:** Instant.

### 👁️ System 2: Imagination (Slow)
- **Mechanism:** `LatentWorldModel` + `TreeSearchPlanner`
- **Usage:** Complex problems, high uncertainty, high risk.
- **Process:**
  1. **Imagine:** Simulate potential actions in latent space ($z_t \to z_{t+1}$).
  2. **Plan:** Search for the path that minimizes predicted energy.
  3. **Act:** Execute the best plan.

---

## Architecture: The Cognitive Stack

### 1. Perception Layer (`vision.py`)
- **Input:** Raw Grid
- **Output:** Object Graph ($G$)

### 2. Intuition Layer (`action_decoder.py`)
- **Input:** Brain State ($z_t$)
- **Output:** Action Logits (Policy)

### 3. Imagination Layer (`imagination.py`) **[NEW]**
- **Input:** State ($z_t$), Action ($a$)
- **Output:** Predicted Next State ($\hat{z}_{t+1}$), Predicted Energy ($\hat{E}$)
- **Role:** The Physics Engine of the Mind.

### 4. Planning Layer (`planner.py`) **[NEW]**
- **Input:** Current State, Imagination Model
- **Output:** Optimal Action ($a^*$)
- **Role:** Tree Search / Lookahead.

### 5. Meta-Cognition Layer (`meta_cognition.py`) **[NEW]**
- **Input:** Entropy, Energy Variance
- **Output:** Control Signal (Switch between System 1 & 2)
- **Role:** The Manager.

---

## Installation & Running

### Prerequisites
```bash
pip install torch>=2.0.0 numpy>=1.24.0
```

### Execution
Run the Meta-Cognitive ASI:
```bash
python main_system.py
```

---

**Architect:** User (The Director)  
**Engineer:** Gemini (Lead Architect)  
**Version:** V4.0 (System 2 Thinking)
