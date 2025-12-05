"""
Meta-Cognition Module (System 2 & 3)

The "Manager" of the mind.
1. Controller: Decides WHEN to think (System 2) and WHEN to act (System 1).
2. MetaLearner: Optimizes HOW the agent learns (System 3).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from collections import deque

class MetaCognitiveController:
    def __init__(self, entropy_threshold: float = 1.5, variance_threshold: float = 0.005):
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        self.energy_history = deque(maxlen=10)
        
    def update_energy(self, energy_value: float):
        self.energy_history.append(energy_value)
        
    def decide_mode(self, entropy: float) -> tuple[str, str]:
        """
        Decide whether to use System 1 (Intuition) or System 2 (Planning).
        """
        # 1. Check Uncertainty (Entropy)
        if entropy > self.entropy_threshold:
            return "SYSTEM_2", f"High Uncertainty (Entropy {entropy:.2f} > {self.entropy_threshold})"
            
        # 2. Check Stability (Energy Variance)
        if len(self.energy_history) >= 5:
            variance = np.var(list(self.energy_history))
            if variance > self.variance_threshold:
                return "SYSTEM_2", f"Unstable Energy (Var {variance:.4f} > {self.variance_threshold})"
        
        return "SYSTEM_1", "Stable & Confident"


class MetaLearner(nn.Module):
    """
    Recursive Self-Improvement System.
    Uses an LSTM to observe the agent's learning trajectory and output dynamic hyperparameters.
    """
    def __init__(self, input_dim=7, hidden_dim=64):
        super(MetaLearner, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM for temporal context
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Heads for distributions (Mu, Sigma)
        self.lr_mu = nn.Linear(hidden_dim, 1)
        self.lr_sigma = nn.Linear(hidden_dim, 1)

        self.cort_mu = nn.Linear(hidden_dim, 1)
        self.cort_sigma = nn.Linear(hidden_dim, 1)

        self.ent_mu = nn.Linear(hidden_dim, 1)
        self.ent_sigma = nn.Linear(hidden_dim, 1)

        # INCREASED LEARNING RATE (Acceleration)
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        last_out = out[:, -1, :]

        # 1. LR Scale (Target mean ~ 1.0)
        lr_mu = torch.sigmoid(self.lr_mu(last_out)) * 4.0 + 0.1
        lr_sigma = torch.nn.functional.softplus(self.lr_sigma(last_out)) + 0.01
        dist_lr = dist.Normal(lr_mu, lr_sigma)
        val_lr = dist_lr.sample()
        val_lr = torch.clamp(val_lr, 0.1, 10.0)

        # 2. Cortisol (Target mean ~ 2.0)
        cort_mu = torch.sigmoid(self.cort_mu(last_out)) * 4.0 + 0.5
        cort_sigma = torch.nn.functional.softplus(self.cort_sigma(last_out)) + 0.01
        dist_cort = dist.Normal(cort_mu, cort_sigma)
        val_cort = dist_cort.sample()
        val_cort = torch.clamp(val_cort, 0.1, 10.0)

        # 3. Entropy (Target mean ~ 0.01)
        ent_mu = torch.sigmoid(self.ent_mu(last_out)) * 0.1
        ent_sigma = torch.nn.functional.softplus(self.ent_sigma(last_out)) + 0.001
        dist_ent = dist.Normal(ent_mu, ent_sigma)
        val_ent = dist_ent.sample()
        val_ent = torch.clamp(val_ent, 0.0, 0.5)

        log_prob = dist_lr.log_prob(val_lr) + dist_cort.log_prob(val_cort) + dist_ent.log_prob(val_ent)

        return (val_lr, val_cort, val_ent), log_prob, hidden

    def store_step(self, log_prob, reward):
        self.saved_log_probs.append(log_prob)
        self.rewards.append(reward)

    def update(self):
        """
        Meta-Update using REINFORCE with PPO-style multi-epoch updates.
        """
        R = 0
        returns = []

        # Calculate Returns
        for r in self.rewards[::-1]:
            R = r + 0.9 * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        if returns.std() > 1e-5:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Standard Update (REINFORCE)
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            # Standard REINFORCE term
            policy_loss.append(-log_prob * R)

        if len(policy_loss) > 0:
            self.optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum()
            loss.backward()

            # Gradient Clipping (Stability)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()

        # Clear buffers
        self.saved_log_probs = []
        self.rewards = []
