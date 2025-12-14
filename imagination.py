"""
Imagination Module (System 2)

The "Inner Eye" of the agent.
Wraps the JEPA Predictor to allow latent space simulation.
Enables the agent to ask: "What if I do this?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from energy import JEPA_Predictor

class LatentWorldModel(nn.Module):
    def __init__(self, predictor: JEPA_Predictor, v_truth: torch.Tensor):
        super().__init__()
        self.predictor = predictor
        self.v_truth = v_truth
        
    def simulate(self, current_state: torch.Tensor, action_id: int, num_actions: int = 6) -> tuple[torch.Tensor, float]:
        """
        Simulate an action in the latent space.
        
        Args:
            current_state: (1, D) Brain state z_t
            action_id: Integer ID of the action to simulate
            num_actions: Total number of possible actions
            
        Returns:
            predicted_next_state: (1, D) z_{t+1}
            predicted_energy: Estimated energy of the resulting state
        """
        # Create action embedding
        action_embedding = F.one_hot(torch.tensor([action_id]), num_classes=num_actions).float()
        # Pad to match state dimension (assuming state_dim=32, action_dim=6 -> pad 26)
        # Note: This padding must match the logic in main_system.py / energy.py
        # In V3.5 energy.py, action_dim was 32. We need to ensure consistency.
        # Assuming predictor expects 32-dim input.
        padding_size = self.predictor.action_dim - num_actions
        if padding_size > 0:
            action_embedding = F.pad(action_embedding, (0, padding_size))
            
        # Predict next state
        with torch.no_grad():
            predicted_next_state, _, _ = self.predictor(current_state, action_embedding)
            
        # Calculate expected energy (Distance to Truth)
        # We don't know the actual next state, so we use the predicted one to estimate "Truth Alignment"
        # Energy = ||z_{t+1} - V_{truth}||^2
        # We cannot estimate "Prediction Error" here because we don't have the real outcome yet.
        # We also assume no "Logical Violation" in imagination (optimistic).
        
        z_target = self.v_truth.unsqueeze(0)
        truth_distance = F.mse_loss(predicted_next_state, z_target)
        
        return predicted_next_state, truth_distance.item()
