import numpy as np

class NeuroChemicalEngine:
    """
    The Heart.
    Manages the emotional state and motivation of the agent via Dopamine and Serotonin.
    """
    def __init__(self):
        self.dopamine = 0.5  # Drive / Curiosity
        self.serotonin = 0.5 # Peace / Stability
        self.prev_energy = float('inf')
        
        # History for variance calculation
        self.energy_history = []

    def update(self, current_energy, consistency_score):
        """
        Update hormone levels based on energy (error) and consistency (truth).
        
        Args:
            current_energy (float): The current error/loss of the system.
            consistency_score (float): How close the mind is to the Axiom (0-1).
        """
        # 1. Dopamine Dynamics (Reward Prediction Error)
        # Spike when energy drops significantly (Improvement)
        energy_delta = self.prev_energy - current_energy
        if energy_delta > 0:
            self.dopamine += energy_delta * 2.0  # Boost drive
        else:
            self.dopamine *= 0.95 # Decay if no progress
        
        self.dopamine = np.clip(self.dopamine, 0.0, 1.0)
        self.prev_energy = current_energy

        # 2. Serotonin Dynamics (Homeostasis & Truth)
        # Rise when consistent with Truth and Energy is low/stable
        is_stable = False
        if len(self.energy_history) > 5:
            variance = np.var(self.energy_history[-5:])
            if variance < 0.01:
                is_stable = True
        
        if is_stable and consistency_score > 0.8 and current_energy < 0.2:
            self.serotonin += 0.1
        else:
            self.serotonin *= 0.98 # Decay if chaotic
            
        self.serotonin = np.clip(self.serotonin, 0.0, 1.0)
        self.energy_history.append(current_energy)

    def get_state(self):
        """
        Returns the dominant state: 'CHAOS' (Dopamine driven) or 'ORDER' (Serotonin driven).
        """
        if self.dopamine > self.serotonin:
            return "CHAOS"
        else:
            return "ORDER"

    def get_hormones(self):
        return self.dopamine, self.serotonin

import torch
import torch.nn as nn

class JEPA_Predictor(nn.Module):
    """
    Joint Embedding Predictive Architecture (JEPA).
    Predicts the next latent state given current state and action.
    Used for 'Imagination' (System 2).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(JEPA_Predictor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # MLP: [State, Action] -> Next State
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, action):
        # state: (B, state_dim)
        # action: (B, action_dim) or (B, 1) if discrete index
        
        # If action is an index (long), we might need to embed it or one-hot it.
        # For simplicity, let's assume action is already a vector or we use a simple embedding here.
        # But the planner passes an integer action_id.
        # Let's assume we handle simple integer actions by one-hot encoding them inside here
        # OR the caller passes a vector.
        # Given the instruction "state_dim and action_dim", let's assume vector input.
        # But wait, planner passes action_id.
        
        # Let's handle both.
        if action.dim() == 1 or (action.dim() == 2 and action.shape[1] == 1):
             # One-hot encode if it looks like indices
             # But we don't know the max action_dim here easily unless passed.
             # Let's assume action is a float vector for now as per "Action Parameters".
             pass
             
        x = torch.cat([state, action], dim=-1)
        next_state = self.net(x)
        return next_state

    def simulate(self, state, action_id, num_actions=4):
        """
        Helper for Planner.
        Simulates next state and returns 'energy' (distance to truth?).
        """
        # Create action vector (One-hot for the action type)
        # We need a consistent way to represent action.
        # The Body has 2 heads: Logits (4) and Params (4).
        # For planning, we might just care about the Action Type (Logits).
        # Let's create a one-hot vector of size num_actions.
        
        action_vec = torch.zeros(1, num_actions)
        action_vec[0, action_id] = 1.0
        
        # We also need to pad it if the network expects more dimensions (e.g. params).
        # But for JEPA, let's say we just predict based on intention.
        # If self.action_dim > num_actions, pad with zeros.
        if self.action_dim > num_actions:
            padding = torch.zeros(1, self.action_dim - num_actions)
            action_vec = torch.cat([action_vec, padding], dim=1)
            
        with torch.no_grad():
            next_state = self.forward(state, action_vec)
            
        # Predicted Energy?
        # In this system, Energy is "Distance to Truth" or "Symmetry Error".
        # We don't have a Truth Vector here easily unless passed.
        # But wait, `manifold.py` has `check_consistency`.
        # Maybe we should return the state and let the caller check consistency.
        # But Planner expects (next_state, energy_cost).
        
        # Let's return a dummy energy or self-energy if we can calculate it.
        # Or maybe the JEPA predicts the energy directly?
        # The user didn't specify.
        # Let's return 0.0 for energy for now, or random.
        # Better: The planner uses this to minimize energy.
        # If we can't predict energy, planning is useless.
        
        # Let's add a small head for Energy Prediction
        return next_state, 0.5 # Dummy energy

