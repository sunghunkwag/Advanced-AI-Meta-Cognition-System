"""
Advanced AI Meta Cognition System
"The Inner Eye"

Main Orchestration Script.
Integrates:
- System 1: Intuition (ActionDecoder)
- System 2: Imagination (LatentWorldModel) & Planning (TreeSearchPlanner)
- Meta-Cognition: The Manager

The Loop:
1. Perceive (Vision)
2. Meta-Cognition Check (Entropy/Variance)
3. IF System 2: Plan (Tree Search) -> Optimal Action
4. IF System 1: React (Policy Sample) -> Action
5. Act (Execute)
6. Learn (Update Brain, Policy, World Model)
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

from soul import get_soul_vectors
from vision import GNNObjectExtractor
from manifold import GraphAttentionManifold
from energy import EnergyFunction, JEPA_Predictor
from automata import ManifoldAutomata
from world import InternalSandbox
from action_decoder import ActionDecoder
from imagination import LatentWorldModel
from planner import TreeSearchPlanner
from meta_cognition import MetaCognitiveController

class AdvancedAgent:
    def __init__(self):
        print("[System] Initializing Cognitive Architecture...")
        
        # 1. Soul
        self.v_identity, self.v_truth, self.v_reject = get_soul_vectors(dim=32)
        
        # 2. Body (System 1)
        self.vision = GNNObjectExtractor(max_objects=5, feature_dim=4)
        self.brain = ManifoldAutomata(state_dim=32, num_heads=4, v_truth=self.v_truth)
        self.action_decoder = ActionDecoder(state_dim=32, num_actions=6)
        self.predictor = JEPA_Predictor(state_dim=32, action_dim=32)
        self.energy_fn = EnergyFunction(lambda_violation=100.0)
        self.world = InternalSandbox()
        
        # 3. Mind (System 2)
        self.imagination = LatentWorldModel(self.predictor, self.v_truth)
        self.planner = TreeSearchPlanner(self.imagination, self.action_decoder, depth=2, num_actions=6)
        self.meta_cognition = MetaCognitiveController(entropy_threshold=2.0, variance_threshold=0.005)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.brain.parameters()) + 
            list(self.action_decoder.parameters()) + 
            list(self.predictor.parameters()), 
            lr=0.001
        )
        
        # Metrics
        self.action_history = []
        self.system2_usage = 0

    def run_cycle(self, step: int):
        print(f"\n{'='*60}")
        print(f"Cycle {step}")
        print(f"{'='*60}")
        
        # --- A. PERCEIVE ---
        if 'grid' in self.world.global_context:
            grid_np = self.world.global_context['grid']
            grid_tensor = torch.tensor(grid_np, dtype=torch.float32)
            node_features, adjacency, num_objects = self.vision(grid_tensor)
            print(f"[Perceive] Observed {num_objects} objects")
        else:
            node_features = torch.zeros(1, 5, 4)
            adjacency = torch.eye(5).unsqueeze(0)
            num_objects = 0
            print("[Perceive] Observed 0 objects (Empty Void)")
            
        # --- B. THINK (Brain Process) ---
        input_state = torch.nn.functional.pad(node_features, (0, 28))
        current_state = self.brain(input_state, adjacency, steps=3)
        z_t = current_state.mean(dim=1) # Global Brain State
        
        # Get Intuition (System 1 Logits)
        action_logits = self.action_decoder(z_t)
        probs = F.softmax(action_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).item()
        
        # --- C. META-COGNITION (Decide Mode) ---
        mode, reason = self.meta_cognition.decide_mode(entropy)
        print(f"[Meta] Mode: {mode} ({reason})")
        
        # --- D. DECIDE (Plan or React) ---
        if mode == "SYSTEM_2":
            self.system2_usage += 1
            # Deliberate using Planner
            best_action, predicted_energy, plan_trace = self.planner.plan(z_t)
            action_id = best_action
            print(f"[System 2] Plan: {plan_trace} -> Energy {predicted_energy:.4f}")
        else:
            # React using Intuition (Stochastic)
            action_id = self.action_decoder.sample_action(action_logits.squeeze(0), deterministic=False)
            print(f"[System 1] Intuition: Selected Action {action_id}")
            
        action_name = self.action_decoder.get_action_name(action_id)
        print(f"[Act] Executing: {action_name}")
        self.action_history.append(action_id)
        
        # --- E. EXECUTE ---
        code = self.action_decoder.get_action_code(action_id)
        output, success = self.world.execute(code)
        
        if not success:
            print(f"[World] Error: {output.strip()[:50]}...")
            
        # --- F. OBSERVE (Next State) ---
        if success and 'grid' in self.world.global_context:
            grid_np = self.world.global_context['grid']
            grid_tensor = torch.tensor(grid_np, dtype=torch.float32)
            next_node_features, next_adjacency, next_num_objects = self.vision(grid_tensor)
        else:
            next_node_features = torch.zeros(1, 5, 4)
            next_adjacency = torch.eye(5).unsqueeze(0)
            next_num_objects = 0
            
        # --- G. LEARN (Update Models) ---
        # 1. Prepare Tensors
        next_input_state = torch.nn.functional.pad(next_node_features, (0, 28))
        next_state = self.brain(next_input_state, next_adjacency, steps=3)
        z_t1 = next_state.mean(dim=1)
        z_target = self.v_truth.unsqueeze(0)
        
        violation = 1.0 if not success else 0.0
        violation_tensor = torch.tensor([violation])
        
        # 2. World Model Loss (JEPA)
        # We MUST train the predictor so System 2 becomes accurate!
        action_embedding = F.one_hot(torch.tensor([action_id]), num_classes=6).float()
        action_embedding = F.pad(action_embedding, (0, 26))
        z_pred = self.predictor(z_t, action_embedding)
        
        pred_error = F.mse_loss(z_pred, z_t1)
        truth_distance = F.mse_loss(z_t1, z_target) * 10.0 # Reinforce Truth Drive
        
        # 1. Boredom Penalty (Punish lack of change)
        state_change = F.mse_loss(z_t, z_t1)
        boredom_penalty = 1.0 / (state_change + 1e-6) * 0.01

        # 2. Void Penalty (Punish empty world)
        void_penalty = 0.0
        if next_num_objects == 0:
            void_penalty = 10.0

        energy = pred_error + truth_distance + violation_tensor.squeeze() * 100.0 + boredom_penalty + void_penalty
        
        # Update Meta-Cognition History
        self.meta_cognition.update_energy(energy.item())
        
        # 3. Policy Loss (REINFORCE)
        log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_prob = log_probs[0, action_id]
        
        # If System 2 was used, we treat its choice as a "Teacher" for System 1?
        # Or just standard REINFORCE based on result?
        # Let's use standard REINFORCE: Minimize Energy.
        policy_loss = selected_log_prob * energy.detach()
        
        # Entropy Bonus (to prevent collapse)
        entropy_bonus = -0.01 * entropy
        
        # EWC
        ewc = self.brain.ewc_loss()
        
        total_loss = energy + ewc - policy_loss + entropy_bonus
        
        print(f"[Heart] Energy: {energy.item():.4f} (PredErr: {pred_error.item():.4f})")
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Intrinsic Crystallization (Same as V3.5.2)
        # ... (Omitted for brevity, but conceptually present)

def main():
    print("="*60)
    print("ADVANCED AI META COGNITION SYSTEM")
    print("System 2 Thinking: Imagination & Planning")
    print("="*60)
    
    agent = AdvancedAgent()
    
    for i in range(300):
        agent.run_cycle(i)
        
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print(f"System 2 Usage: {agent.system2_usage}/300 cycles")
    print("="*60)

if __name__ == "__main__":
    main()
