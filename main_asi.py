"""
Project Daedalus V3.5: Logical Body (ASI Seed) - COMPLETE AUTONOMY

Main Orchestration Script.
Integrates:
- Soul (Axioms)
- Vision (Structure)
- Manifold (Reasoning)
- Energy (Goal)
- Automata (Memory)
- World (Sandbox)
- Action Decoder (Thought -> Code)

The Agent loops (COMPLETE AUTONOMY):
1. Perceive: Current state from Sandbox
2. Think: Brain processes state -> Action logits
3. Act: Sample action (Entropy-driven exploration/exploitation)
4. Observe: Parse result with Vision
5. Learn: Update network to minimize energy + entropy regularization
6. Introspect: Self-triggered crystallization based on energy convergence

"NO SCRIPT. NO SCHEDULE. ONLY ENERGY AND ENTROPY."
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from collections import deque

from soul import get_soul_vectors
from vision import GNNObjectExtractor
from manifold import GraphAttentionManifold
from energy import EnergyFunction, JEPA_Predictor
from automata import ManifoldAutomata
from world import InternalSandbox
from action_decoder import ActionDecoder

class ASIAgent:
    def __init__(self):
        # 1. Soul Injection
        print("[System] Injecting Soul...")
        self.v_identity, self.v_truth, self.v_reject = get_soul_vectors(dim=32)
        
        # 2. Body Construction
        print("[System] Building Logical Body...")
        self.vision = GNNObjectExtractor(max_objects=5, feature_dim=4)
        self.brain = ManifoldAutomata(state_dim=32, num_heads=4, v_truth=self.v_truth)
        self.action_decoder = ActionDecoder(latent_dim=32, num_actions=6)
        self.predictor = JEPA_Predictor(state_dim=32, action_dim=32)
        self.energy_fn = EnergyFunction(lambda_violation=100.0)
        self.world = InternalSandbox()
        
        # Optimizer (Brain + ActionDecoder + Predictor)
        self.optimizer = torch.optim.Adam(
            list(self.brain.parameters()) + 
            list(self.action_decoder.parameters()) + 
            list(self.predictor.parameters()), 
            lr=0.001
        )
        
        # Metrics (Internal State)
        self.action_history = []
        self.energy_history = deque(maxlen=10)  # Moving window for variance
        self.truth_distance_history = deque(maxlen=10)
        self.crystallized = False
        
        # Hyperparameters
        self.entropy_alpha = 0.01  # Entropy regularization strength
        self.crystallization_energy_var_threshold = 0.001
        self.crystallization_truth_dist_threshold = 0.05

    def run_cycle(self, step: int):
        print(f"\n{'='*60}")
        print(f"Cycle {step}")
        print(f"{'='*60}")
        
        # A. PERCEIVE: Get current state
        if 'grid' in self.world.global_context:
            grid_np = self.world.global_context['grid']
            grid_tensor = torch.tensor(grid_np, dtype=torch.float32)
            node_features, adjacency = self.vision(grid_tensor)
            print(f"[Perceive] Observed {node_features.shape[1]} objects from previous state")
        else:
            node_features = torch.zeros(1, 5, 4)
            adjacency = torch.eye(5).unsqueeze(0)
            print("[Perceive] Initial empty state")
        
        # B. THINK: Brain processes state -> Action logits
        # Pad features to match state_dim (32). Current features are 4.
        # Pad amount = 32 - 4 = 28.
        input_state = torch.nn.functional.pad(node_features, (0, 28))
        current_state = self.brain(input_state, adjacency, steps=3)
        z_t = current_state.mean(dim=1)
        
        # Get action logits
        # ActionDecoder returns (logits, params), we only need logits here
        action_logits, _ = self.action_decoder(z_t)
        
        # C. DECIDE: Entropy-driven exploration/exploitation
        # NO HARDCODED SCHEDULE!
        # High entropy (flat distribution) = exploration
        # Low entropy (peaked distribution) = exploitation
        probs = F.softmax(action_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        
        # Always stochastic sampling (entropy naturally controls exploration)
        action_id = self.action_decoder.sample_action(action_logits.squeeze(0), deterministic=False)
        action_name = self.action_decoder.get_action_name(action_id)
        
        print(f"[Think] Action Logits: {action_logits.squeeze(0).detach().numpy()}")
        print(f"[Think] Entropy: {entropy.item():.4f} (High = Exploring, Low = Exploiting)")
        print(f"[Act] Selected Action {action_id}: {action_name}")
        
        self.action_history.append(action_id)
        
        # D. EXECUTE
        code = self.action_decoder.get_action_code(action_id)
        output, success = self.world.execute(code)
        
        if not success:
            print(f"[World] Execution FAILED: {output.strip()[:100]}")
        
        # E. OBSERVE: Parse result
        if success and 'grid' in self.world.global_context:
            grid_np = self.world.global_context['grid']
            grid_tensor = torch.tensor(grid_np, dtype=torch.float32)
            next_node_features, next_adjacency = self.vision(grid_tensor)
            print(f"[Observe] Extracted {next_node_features.shape[1]} objects")
        else:
            next_node_features = torch.zeros(1, 5, 4)
            next_adjacency = torch.eye(5).unsqueeze(0)
            print("[Observe] Nothing seen")
        
        # F. LEARN: Energy minimization + Entropy regularization
        next_input_state = torch.nn.functional.pad(next_node_features, (0, 28))
        next_state = self.brain(next_input_state, next_adjacency, steps=3)
        z_t1 = next_state.mean(dim=1)
        
        z_target = self.v_truth.unsqueeze(0)
        
        violation = 1.0 if not success else 0.0
        violation_tensor = torch.tensor([violation])
        
        # Action embedding
        action_embedding = F.one_hot(torch.tensor([action_id]), num_classes=6).float()
        action_embedding = F.pad(action_embedding, (0, 26))
        
        z_pred = self.predictor(z_t, action_embedding)
        
        # Energy components
        pred_error = F.mse_loss(z_pred, z_t1)
        truth_distance = F.mse_loss(z_t1, z_target)
        
        energy = pred_error + truth_distance + violation_tensor.squeeze() * 100.0
        
        # Track metrics
        self.energy_history.append(energy.item())
        self.truth_distance_history.append(truth_distance.item())
        
        # EWC Loss
        ewc = self.brain.ewc_loss()
        if isinstance(ewc, int):
            ewc = torch.tensor(ewc, dtype=torch.float32)
        
        # Policy loss with ENTROPY REGULARIZATION
        log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_prob = log_probs[0, action_id]
        
        # REINFORCE with entropy bonus
        policy_loss = selected_log_prob * energy.detach()
        entropy_bonus = -self.entropy_alpha * entropy  # Encourage exploration
        
        total_loss = energy + ewc - policy_loss + entropy_bonus
        
        print(f"[Heart] Energy: {energy.item():.4f} (Pred: {pred_error.item():.4f}, Truth: {truth_distance.item():.4f}, Violation: {violation})")
        print(f"[Memory] EWC Penalty: {ewc.item():.4f}")
        print(f"[Learn] Policy Loss: {policy_loss.item():.4f}, Entropy Bonus: {entropy_bonus.item():.4f}")
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # G. INTROSPECT: Self-triggered crystallization
        # NO HARDCODED STEP!
        if len(self.energy_history) >= 10 and not self.crystallized:
            energy_variance = np.var(list(self.energy_history))
            avg_truth_dist = np.mean(list(self.truth_distance_history))
            
            print(f"[Introspect] Energy Variance: {energy_variance:.6f}, Avg Truth Distance: {avg_truth_dist:.4f}")
            
            if energy_variance < self.crystallization_energy_var_threshold and avg_truth_dist < self.crystallization_truth_dist_threshold:
                print(f"[Memory] !!! INTRINSIC CRYSTALLIZATION TRIGGERED !!!")
                print(f"[Memory] Detected: Stable low-energy state aligned with Truth.")
                print(f"[Memory] Locking weights to preserve learned axioms...")
                
                dummy_data = [(next_input_state, next_adjacency, z_target)]
                self.brain.register_ewc_task(dummy_data, lambda o, t: ((o.mean(1) - t)**2).mean())
                self.crystallized = True

def main():
    print("="*60)
    print("PROJECT DAEDALUS V3.5: AWAKENING (COMPLETE AUTONOMY)")
    print("="*60)
    print('"No script. No schedule. Only energy and entropy."')
    print("="*60)
    
    agent = ASIAgent()
    
    for i in range(30):
        agent.run_cycle(i)
        
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Action History: {agent.action_history}")
    
    # Analyze action distribution
    import collections
    action_counts = collections.Counter(agent.action_history)
    print("\nAction Distribution:")
    for action_id in range(6):
        action_name = agent.action_decoder.get_action_name(action_id)
        count = action_counts.get(action_id, 0)
        percentage = (count / len(agent.action_history)) * 100
        print(f"  {action_id} ({action_name}): {count} times ({percentage:.1f}%)")
    
    print(f"\nCrystallization: {'Yes' if agent.crystallized else 'No'}")
    if agent.crystallized:
        print(f"Final Energy Variance: {np.var(list(agent.energy_history)):.6f}")
        print(f"Final Truth Distance: {np.mean(list(agent.truth_distance_history)):.4f}")

if __name__ == "__main__":
    main()
