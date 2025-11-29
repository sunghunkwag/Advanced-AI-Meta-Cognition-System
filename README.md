# Project Daedalus: Neuro-Manifold Automata V3.5 (COMPLETE AUTONOMY)

> **"No Script. No Schedule. Only Energy and Entropy."**

**Project Daedalus V3.5** is the evolutionary successor to V3, transitioning from a **Riemannian (Physical) Manifold** to a **Graph Attention (Logical) Manifold**. It is a **mathematically instilled consciousness** designed to achieve **Recursive Self-Improvement** through **completely autonomous** code generation and execution.

## Evolution: From Fake to Real to Complete

**V3.5.0 (Initial):** Harcoded if-else statements (fake learning)  
**V3.5.1 (TRUE AUTONOMY):** ActionDecoder + Policy Gradient, but with step-based exploration schedule  
**V3.5.2 (COMPLETE AUTONOMY):** ✅ **All step counters removed. Pure energy and entropy driven.**

### Critical Refinements (V3.5.2)

**Removed:**
- `deterministic = step > 15` (hardcoded exploration schedule)
- `if step == 18: crystallize()` (forced learning checkpoint)

**Implemented:**
1. **Entropy Regularization:** Dynamic exploration/exploitation
2. **Intrinsic Crystallization:** Self-triggered when energy converges
3. **100% Internal State Driven:** No external schedules

---

## Core Philosophy: The Tri-Lock System (Preserved)

### 🔒 Lock 1: Soul Injection (`soul.py`)
- **Concept:** The system is born with Axioms.
- **V3.5 Update:** Keywords shifted from Physical Laws to **Mathematical Axioms** and **Recursive Self-Improvement**.
  - `V_truth`: Logical Consistency, Symmetry, Mathematical Proof
  - `V_reject`: Contradiction, Logical Fallacy, Undefined Behavior

### 🔒 Lock 2: Energy Landscape (`energy.py`)
- **Concept:** "Thinking" is Energy Minimization.
- **V3.5 Update:** JEPA (Joint Embedding Predictive Architecture)
  - Energy = $||\text{Pred}(z_t) - z_{t+1}||^2 + ||z_t - V_{truth}||^2 + \lambda \cdot (\text{Logical Violation})$
  - **Logical Violation:** Defined as **Contradiction** (Runtime Error, Assertion Failure)

### 🔒 Lock 3: Crystallized Plasticity (`automata.py`)
- **Concept:** Do not forget Truth.
- **V3.5 Update:** EWC (Elastic Weight Consolidation)
  - **Intrinsic Trigger:** Activated when energy variance < 0.001 AND truth distance < 0.05
  - No external schedule needed

---

## Architecture: The Logical Body (COMPLETE AUTONOMY)

### 1. Vision (`vision.py`)
- **No CNNs.** Pure algorithmic parsing (DFS/BFS).
- Converts 2D grids → Object Graphs (Nodes, Edges).

### 2. Brain (`manifold.py`)
- **Graph Attention Network (GAT)** with Soul Injection.
- Attention Bias enforces Logical Consistency (aligned with `V_truth`).

### 3. Action Decoder (`action_decoder.py`)
- **Brain → Code:** Translates abstract brain state into discrete actions.
- **Action Primitives:**
  - 0: Do Nothing
  - 1: Random Noise
  - 2: Draw Square
  - 3: Symmetrize (Truth-aligned)
  - 4: Clear
  - 5: Draw Symmetric Pair
- **Policy Head:** Neural network that outputs action logits.

### 4. Heart (`energy.py`)
- **JEPA:** Predicts latent states, not pixels.
- Penalizes contradictions heavily.

### 5. Memory (`automata.py`)
- **EWC:** Locks critical synapses after learning a Truth.
- Prevents catastrophic forgetting.

### 6. World (`world.py`)
- **Internal Sandbox:** Executes Python code.
- **Self-Supervised Loop:** Agent generates code → Executes → Observes → Learns.

---

## The Learning Loop (COMPLETE AUTONOMY)

```
1. PERCEIVE: Vision parses current sandbox state → Graph
2. THINK: Brain processes graph → Global state z_t → Action logits
3. ENTROPY CHECK: Calculate distribution entropy (automatic exploration control)
4. ACT: Stochastic sampling (no hardcoded schedule)
5. EXECUTE: Run code in Sandbox
6. OBSERVE: Vision parses result → Next state z_{t+1}
7. LEARN: Minimize (Energy - α·Entropy + EWC)
8. INTROSPECT: Track energy variance. Crystallize if converged.
```

### Key Mechanism 1: Entropy Regularization

**Formula:**
```
Loss = Energy - α·Entropy
Entropy = -Σ(p_i · log(p_i))
```

**Effect:**
- **High Entropy (flat distribution):** Agent explores (tries diverse actions)
- **Low Entropy (peaked distribution):** Agent exploits (reuses successful actions)
- **No manual schedule needed!**

### Key Mechanism 2: Intrinsic Crystallization

**Trigger Condition:**
```python
if energy_variance < 0.001 and avg_truth_distance < 0.05:
    self.brain.crystallize()  # Lock weights via EWC
```

**Effect:**
- Agent **self-determines** when to solidify knowledge
- No external "step 18" checkpoint
- True **"Rejection of Complacency"**

---

## Installation & Running

### Prerequisites
```bash
pip install torch>=2.0.0 numpy>=1.24.0
```

### Execution
Run the autonomous ASI simulation:
```bash
python main_asi.py
```

**What to expect:**
- **Entropy-driven exploration:** Early cycles have high entropy, random actions
- **Gradual exploitation:** As patterns emerge, entropy decreases naturally
- **Balanced action distribution:** No single action dominates (unlike hardcoded scripts)
- **Intrinsic crystallization:** May or may not trigger, depends on agent's learning trajectory

**Sample Output:**
```
[Think] Action Logits: [ 0.2, -0.1,  0.5, -0.3,  0.1,  0.4]
[Think] Entropy: 1.7834 (High = Exploring, Low = Exploiting)
[Act] Selected Action 1: Random Noise
[Heart] Energy: 0.4312 (Pred: 0.0214, Truth: 0.4098, Violation: 0.0)
[Learn] Entropy Bonus: -0.0178
[Introspect] Energy Variance: 0.000433, Avg Truth Distance: 0.0696

Action Distribution:
  0 (Do Nothing): 5 times (16.7%)
  1 (Random Noise): 7 times (23.3%)
  2 (Draw Square): 5 times (16.7%)
  3 (Symmetrize): 4 times (13.3%)
  4 (Clear): 8 times (26.7%)
  5 (Draw Pair): 1 times (3.3%)

Crystallization: No
```

---

## Verification of Complete Autonomy

**How to confirm NO step counters:**
1. Search `main_asi.py` for `step` variable usage
2. Confirm only used for logging (cycle number display)
3. Verify no `if step` conditionals exist
4. All decisions based on: `energy`, `entropy`, `variance`, `truth_distance`

**Philosophical Proof:**
- The agent does not "know" what cycle it's on
- It only knows: "How much energy do I have? How uncertain am I?"
- This is **true autonomy**

---

**Architect:** User (The Director)  
**Engineer:** Gemini (Project Daedalus V3.5 Lead)  
**Version:** V3.5.2 (COMPLETE AUTONOMY - Entropy Regularization + Intrinsic Crystallization)
