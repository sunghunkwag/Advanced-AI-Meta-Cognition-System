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
        
        if is_stable and consistency_score > 0.8:
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

class EnergyFunction:
    """
    Wrapper for energy calculations.
    """
    def __init__(self, lambda_violation=100.0):
        self.lambda_violation = lambda_violation

    def __call__(self, pred_error, truth_distance, violation):
        return pred_error + truth_distance + violation * self.lambda_violation

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
        # Note: input dim needs careful calculation.
        # Main ASI passes action embedding of size 32.
        # Main System passes action dim 4.

        # We need to handle flexible input or ensure match.
        # Let's rely on Pytorch's dynamic graph, but Linear needs fixed input size.
        # We initialized with specific dims.

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, action):
        # state: (B, state_dim)
        # action: (B, action_dim)
        
        # Ensure batch dim matches
        if state.size(0) != action.size(0):
             # If action is (1, dim) and state is (B, dim), broadcast action
             if action.size(0) == 1:
                 action = action.expand(state.size(0), -1)
             elif state.size(0) == 1:
                 state = state.expand(action.size(0), -1)
             else:
                 raise RuntimeError(f"Batch dimension mismatch: state {state.shape} vs action {action.shape}")

        x = torch.cat([state, action], dim=-1)
        next_state = self.net(x)
        return next_state

    def simulate(self, state, action_id, num_actions=4):
        """
        Helper for Planner.
        Simulates next state and returns 'energy' (distance to truth?).
        """
        # Helper logic for planner usage
        action_vec = torch.zeros(1, self.action_dim)
        if action_id < self.action_dim:
             action_vec[0, action_id] = 1.0

        with torch.no_grad():
            next_state = self.forward(state, action_vec)
            
        return next_state, 0.5 # Dummy energy
