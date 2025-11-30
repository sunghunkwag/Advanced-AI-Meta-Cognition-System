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
    def __init__(self, predictor: JEPA_Predictor, v_truth: torch.Tensor, energy_fn=None):
        super().__init__()
        self.predictor = predictor
        self.v_truth = v_truth
        self.energy_fn = energy_fn
        
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
        # z_pred = Predictor(z_t, action)
        with torch.no_grad():
            predicted_next_state = self.predictor(current_state, action_embedding)
            
        # Calculate expected energy (Distance to Truth)
        # We don't know the actual next state, so we use the predicted one to estimate "Truth Alignment"
        # Energy = ||z_{t+1} - V_{truth}||^2
        # We cannot estimate "Prediction Error" here because we don't have the real outcome yet.
        # We also assume no "Logical Violation" in imagination (optimistic).
        
        # Use Neuro-Chemical Core if available
        if self.energy_fn:
            serotonin = self.energy_fn.compute_serotonin(predicted_next_state, self.v_truth)
            effective_boredom = self.energy_fn.compute_boredom(current_state, predicted_next_state, serotonin)

            # Planner minimizes "Energy" (Cost).
            # Reward = Serotonin - Boredom (Ignoring Dopamine for single-step lookahead)
            # Cost = -Reward = Boredom - Serotonin
            # Add Truth Distance as a proxy for "Base Energy" if Serotonin is just a bonus?
            # Serotonin IS based on Truth Distance (1/dist).
            # So maximizing Serotonin IS minimizing Truth Distance.
            # So we can just use Cost = effective_boredom - serotonin.
            # But Serotonin is small (0-100), Boredom can be huge.
            # Let's match main_system logic: Total Reward = Sero - Bore.
            # Cost = Bore - Sero.
            estimated_energy = effective_boredom - serotonin
        else:
            # Fallback (should not happen with correct init)
            z_target = self.v_truth.unsqueeze(0)
            truth_distance = F.mse_loss(predicted_next_state, z_target) * 10.0
            state_change = F.mse_loss(current_state, predicted_next_state)
            boredom_penalty = 1.0 / (state_change + 1e-6) * 0.01
            estimated_energy = truth_distance + boredom_penalty
        
        if isinstance(estimated_energy, torch.Tensor):
            estimated_energy = estimated_energy.item()

        return predicted_next_state, estimated_energy
