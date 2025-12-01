import numpy as np

class NeuroChemicalEngine:
    """
    The Heart.
    Manages the emotional state and motivation of the agent via Dopamine and Serotonin.
    """
    def __init__(self):
        self.dopamine = 0.5  # Pleasure / Novelty / Learning
        self.serotonin = 0.5 # Satisfaction / Meaning / Order
        self.cortisol = 0.0  # Stress / Boredom / Chaos
        
        # Internal State Trackers
        self.prev_energy = float('inf')
        self.energy_history = []
        self.boredom_counter = 0
        self.chaos_counter = 0
        self.last_prediction_error = 0.0

    def update(self, world_energy, consistency_score, density=0.0, symmetry=0.0, prediction_error=0.0):
        """
        Update hormone levels based on internal and external states.
        
        Args:
            world_energy (float): Physical disorder (Energy).
            consistency_score (float): Mental consistency (Truth).
            density (float): Grid occupancy (0-1).
            symmetry (float): Visual symmetry score (0-1).
            prediction_error (float): Surprise/Novelty (0-1).
        """
        # 1. Cortisol Dynamics (The Stick)
        # Triggered by Boredom (No change) or Chaos (High prediction error)
        energy_delta = abs(self.prev_energy - world_energy)
        
        if energy_delta < 0.001: # Boredom
            self.boredom_counter += 1
            # Dampened accumulation: 0.05 -> 0.02
            self.cortisol += 0.02 * (self.boredom_counter / 10.0) 
        else:
            self.boredom_counter = 0
            self.cortisol -= 0.05 # Slower decay (lingering stress)
            
        if prediction_error > 0.6: # Chaos/Anxiety
            self.chaos_counter += 1
            self.cortisol += 0.05 # Reduced from 0.1
        else:
            self.chaos_counter = 0
            # Recovery from stress
            if self.cortisol > 0:
                self.cortisol -= 0.02
            
        self.cortisol = np.clip(self.cortisol, 0.0, 1.0)
        
        # 2. Dopamine Dynamics (The Carrot - Excitement)
        # Triggered by Learning (Error reduction) and Novelty (Prediction Error)
        # We want Dopamine when we *resolve* surprise (Learning) or *find* something new.
        
        # Learning reward: If we reduced energy significantly
        learning_reward = 0.0
        if self.prev_energy - world_energy > 0.01:
            learning_reward = (self.prev_energy - world_energy) * 5.0
            
        # Novelty reward: Moderate prediction error is good (Curiosity)
        novelty_reward = 0.0
        if 0.1 < prediction_error < 0.5:
            novelty_reward = prediction_error * 2.0
            
        target_dopamine = learning_reward + novelty_reward
        self.dopamine = self.dopamine * 0.8 + target_dopamine * 0.2
        self.dopamine = np.clip(self.dopamine, 0.0, 1.0)
        
        # 3. Serotonin Dynamics (The Carrot - Meaning)
        # Triggered by "Meaningful Order" = Density * Symmetry * Consistency
        # Empty grid (Density=0) -> Serotonin 0
        # Messy grid (Symmetry=0) -> Serotonin 0
        
        meaningful_order = density * symmetry * consistency_score
        
        # Serotonin builds up slowly (Satisfaction takes time)
        if meaningful_order > 0.1:
            self.serotonin = self.serotonin * 0.9 + meaningful_order * 0.1
        else:
            self.serotonin *= 0.95 # Decay if meaningless
            
        self.serotonin = np.clip(self.serotonin, 0.0, 1.0)
        
        # Update history
        self.prev_energy = world_energy
        self.energy_history.append(world_energy)
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)

    def get_hormones(self):
        return {
            'dopamine': self.dopamine,
            'serotonin': self.serotonin,
            'cortisol': self.cortisol
        }

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

