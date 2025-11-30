"""Enhanced Neuro-Chemical Engine with improved JEPA predictor.

Extends the original energy.py with better world model prediction
and integrated energy forecasting.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class NeuroChemicalEngine:
    """Enhanced Heart with configurable parameters.
    
    Manages emotional state via Dopamine (drive) and Serotonin (peace).
    """
    
    def __init__(self, config=None):
        if config is None:
            # Default values
            self.dopamine = 0.5
            self.serotonin = 0.5
            self.dopamine_boost_factor = 2.0
            self.dopamine_decay = 0.95
            self.serotonin_boost = 0.1
            self.serotonin_decay = 0.98
            self.stability_threshold = 0.01
            self.stability_window = 5
            self.consistency_threshold = 0.8
            self.energy_threshold = 0.2
        else:
            # Load from config
            self.dopamine = config.initial_dopamine
            self.serotonin = config.initial_serotonin
            self.dopamine_boost_factor = config.dopamine_boost_factor
            self.dopamine_decay = config.dopamine_decay
            self.serotonin_boost = config.serotonin_boost
            self.serotonin_decay = config.serotonin_decay
            self.stability_threshold = config.stability_variance_threshold
            self.stability_window = config.stability_history_window
            self.consistency_threshold = config.consistency_threshold
            self.energy_threshold = config.energy_threshold
        
        self.prev_energy = float('inf')
        self.energy_history = []

    def update(self, current_energy: float, consistency_score: float):
        """Update hormone levels based on energy and consistency.
        
        Args:
            current_energy: Current system error/loss
            consistency_score: Alignment with truth axioms (0-1)
        """
        # Dopamine: Reward Prediction Error (RPE)
        energy_delta = self.prev_energy - current_energy
        if energy_delta > 0:
            self.dopamine += energy_delta * self.dopamine_boost_factor
        else:
            self.dopamine *= self.dopamine_decay
        
        self.dopamine = np.clip(self.dopamine, 0.0, 1.0)
        self.prev_energy = current_energy

        # Serotonin: Homeostasis & Truth Alignment
        is_stable = self._check_stability()
        
        if (is_stable and 
            consistency_score > self.consistency_threshold and 
            current_energy < self.energy_threshold):
            self.serotonin += self.serotonin_boost
        else:
            self.serotonin *= self.serotonin_decay
            
        self.serotonin = np.clip(self.serotonin, 0.0, 1.0)
        self.energy_history.append(current_energy)
        
        # Keep history bounded
        if len(self.energy_history) > 100:
            self.energy_history = self.energy_history[-100:]

    def _check_stability(self) -> bool:
        """Check if system is in stable state."""
        if len(self.energy_history) < self.stability_window:
            return False
        
        recent = self.energy_history[-self.stability_window:]
        variance = np.var(recent)
        return variance < self.stability_threshold

    def get_state(self) -> str:
        """Return dominant state: CHAOS or ORDER."""
        return "CHAOS" if self.dopamine > self.serotonin else "ORDER"

    def get_hormones(self) -> Tuple[float, float]:
        """Return current hormone levels."""
        return self.dopamine, self.serotonin


class ImprovedJEPA(nn.Module):
    """Enhanced Joint Embedding Predictive Architecture.
    
    Predicts next latent state AND energy cost for planning.
    Supports both discrete action indices and continuous parameters.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State predictor
        self.state_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Energy predictor (for planning)
        self.energy_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Energy normalized to [0, 1]
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: predict next state and energy.
        
        Args:
            state: (B, state_dim) current latent state
            action: (B, action_dim) action vector
            
        Returns:
            next_state: (B, state_dim) predicted next state
            energy: (B, 1) predicted energy cost
        """
        x = torch.cat([state, action], dim=-1)
        next_state = self.state_predictor(x)
        energy = self.energy_predictor(x)
        return next_state, energy
    
    def simulate(self, state: torch.Tensor, action_id: int, 
                 num_actions: int = 4) -> Tuple[torch.Tensor, float]:
        """Simulate action and return predicted state and energy.
        
        Used by planner for tree search.
        
        Args:
            state: (1, state_dim) current state
            action_id: discrete action index
            num_actions: total number of discrete actions
            
        Returns:
            next_state: (1, state_dim) predicted state
            energy: predicted energy cost (scalar)
        """
        # One-hot encode action
        action_vec = torch.zeros(1, num_actions, device=state.device)
        action_vec[0, action_id] = 1.0
        
        # Pad if needed
        if self.action_dim > num_actions:
            padding = torch.zeros(1, self.action_dim - num_actions, device=state.device)
            action_vec = torch.cat([action_vec, padding], dim=1)
        
        with torch.no_grad():
            next_state, energy = self.forward(state, action_vec)
        
        return next_state, energy.item()
    
    def train_step(self, state: torch.Tensor, action: torch.Tensor,
                   next_state_target: torch.Tensor, energy_target: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """Training step for world model learning.
        
        Args:
            state: current states
            action: actions taken
            next_state_target: actual next states
            energy_target: actual energy values
            optimizer: optimizer instance
            
        Returns:
            state_loss: prediction loss for states
            energy_loss: prediction loss for energy
        """
        next_state_pred, energy_pred = self.forward(state, action)
        
        state_loss = nn.functional.mse_loss(next_state_pred, next_state_target)
        energy_loss = nn.functional.mse_loss(energy_pred, energy_target)
        
        total_loss = state_loss + energy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return state_loss.item(), energy_loss.item()
