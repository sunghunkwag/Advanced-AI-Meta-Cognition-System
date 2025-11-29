"""
Planner Module (V4.0 - System 2)

The "Strategist" of the agent.
Performs Tree Search (Lookahead) using the Latent World Model (Imagination).
Finds the action that minimizes predicted energy over a future horizon.
"""

import torch
import math

class TreeSearchPlanner:
    def __init__(self, world_model, action_decoder, depth: int = 2, num_actions: int = 6):
        self.world_model = world_model
        self.action_decoder = action_decoder
        self.depth = depth
        self.num_actions = num_actions
        
    def plan(self, current_state: torch.Tensor) -> tuple[int, float, str]:
        """
        Perform lookahead search to find the best action.
        
        Args:
            current_state: (1, D) Current brain state
            
        Returns:
            best_action_id: The optimal action to take now
            min_energy: The predicted energy of the best path
            plan_description: String describing the thought process
        """
        best_action_id = -1
        min_energy = float('inf')
        best_path = []
        
        # Simple Depth-First Search (or Breadth-First) for short horizons
        # For depth=2, we check Action1 -> Action2
        
        print(f"[System 2] Deliberating (Depth {self.depth})...")
        
        # We will implement a recursive search
        best_action_id, min_energy, path_trace = self._search(current_state, current_depth=0)
        
        return best_action_id, min_energy, path_trace

    def _search(self, state: torch.Tensor, current_depth: int) -> tuple[int, float, str]:
        # Base case: if we reached max depth, return 0 energy (or heuristic)
        # Actually, we want to minimize the CUMULATIVE or FINAL energy.
        # Let's minimize the energy of the FINAL state in the horizon.
        
        if current_depth == self.depth:
            return -1, 0.0, "" # Action doesn't matter at leaf
            
        best_action = -1
        min_path_energy = float('inf')
        best_trace = ""
        
        # Try all actions
        for action_id in range(self.num_actions):
            # 1. Imagine consequences
            next_state, energy_cost = self.world_model.simulate(state, action_id, self.num_actions)
            
            # 2. Recurse
            if current_depth < self.depth - 1:
                _, future_energy, sub_trace = self._search(next_state, current_depth + 1)
                total_energy = energy_cost + future_energy # Cumulative energy? Or just final?
                # Let's use a discounted sum or just the minimum energy found along the path.
                # For simplicity in V4.0: Minimize the Energy of the resulting state + future.
                # Heuristic: Energy is "Distance to Truth". We want to get closer.
                # So we want the path that leads to the closest state to Truth.
                # Let's take the energy of the *final* state as the metric.
                # But `energy_cost` returned by simulate IS the distance to truth of next_state.
                # So we want to minimize the minimum energy encountered or the final one.
                # Let's minimize the energy at the end of the step.
                
                # Let's use: Cost = Immediate Energy + Future Energy
                total_energy = energy_cost + 0.9 * future_energy
                
                trace = f"[{action_id}]->{sub_trace}"
            else:
                # Leaf node
                total_energy = energy_cost
                trace = f"[{action_id}]"
            
            if total_energy < min_path_energy:
                min_path_energy = total_energy
                best_action = action_id
                best_trace = trace
                
        return best_action, min_path_energy, best_trace
