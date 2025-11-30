"""
Energy Module (V3.5 - Logical Body)

Goal: "Understanding" as Energy Minimization.
Method: JEPA (Joint Embedding Predictive Architecture).
Energy Function: E = Prediction Error + Logical Violation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class JEPA_Predictor(nn.Module):
    """
    Predicts the next latent state given the current state and an action/context.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        z_t: (B, D)
        action: (B, A) - Encoded action or context
        Returns: z_pred (B, D)
        """
        inp = torch.cat([z_t, action], dim=-1)
        return self.net(inp)

class EnergyFunction(nn.Module):
    """
    Computes the Energy (Loss) of the system.
    """
    def __init__(self, lambda_violation: float = 100.0):
        super().__init__()
        self.lambda_violation = lambda_violation

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor, violation_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes Energy.
        z_pred: (B, D) - Predicted next state
        z_target: (B, D) - Actual next state (Embedding of the observation)
        violation_mask: (B,) - 1.0 if Logical Violation (Contradiction/Error) occurred, 0.0 otherwise.
        """
        # 1. Prediction Error (Latent Distance)
        # We want the prediction to match the reality.
        pred_error = F.mse_loss(z_pred, z_target, reduction='none').mean(dim=-1) # (B,)

        # 2. Logical Violation Penalty
        # If the action led to a contradiction (e.g. syntax error, assertion fail),
        # we impose a massive energy penalty.
        # This forces the agent to avoid "Thinking" in invalid ways.
        violation_energy = self.lambda_violation * violation_mask

        total_energy = pred_error + violation_energy
        
        return total_energy.mean()

    # =========================================================================
    # NEURO-CHEMICAL CORE (Dopamine & Serotonin)
    # =========================================================================

    def compute_dopamine(self, prev_energy: float, current_energy: float) -> float:
        """
        Dopamine: The Drive.
        Reward based on rate of energy decrease.
        """
        if prev_energy is None:
            return 0.0
        return max(0.0, prev_energy - current_energy)

    def compute_serotonin(self, z_t: torch.Tensor, v_truth: torch.Tensor) -> float:
        """
        Serotonin: The Peace.
        Reward based on closeness to Truth.
        """
        # Ensure v_truth has batch dim if needed, or broadcast
        if v_truth.dim() == 1:
            target = v_truth.unsqueeze(0)
        else:
            target = v_truth

        dist = F.mse_loss(z_t, target)
        return 1.0 / (dist.item() + 1e-4)

    def compute_boredom(self, z_t: torch.Tensor, z_t1: torch.Tensor, serotonin: float) -> float:
        """
        Boredom: The Engine.
        Regulated by Serotonin Brake.
        """
        # Raw Boredom: Punish lack of state change
        state_change = F.mse_loss(z_t, z_t1).item()
        raw_boredom = 1.0 / (state_change + 1e-6) * 0.01 # Scaling factor from previous logic

        # Serotonin Brake
        # High Serotonin -> High Tanh -> Low Brake Factor -> Low Boredom
        brake = 1.0 - math.tanh(serotonin)

        effective_boredom = raw_boredom * brake
        return effective_boredom
