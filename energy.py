import numpy as np
import torch
import torch.nn as nn


class NeuroChemicalEngine:
    """
    The Heart.
    Manages the emotional state and motivation of the agent via Dopamine and Serotonin.
    """

    def __init__(self):
        self.dopamine = 0.5  # Pleasure / Novelty / Learning
        self.serotonin = 0.5  # Satisfaction / Meaning / Order
        self.cortisol = 0.0  # Stress / Boredom / Chaos
        self.dopamine_gain = 1.0
        
        # Internal State Trackers
        self.prev_energy = float("inf")
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
            # Increased accumulation rate
            self.cortisol += 0.03 * (self.boredom_counter / 10.0)
        else:
            self.boredom_counter = 0
            # Slower decay to allow stress to linger
            self.cortisol -= 0.02
            
        if prediction_error > 0.6: # Chaos/Anxiety
            self.chaos_counter += 1
            self.cortisol += 0.05  # Reduced from 0.1
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
        self.dopamine = self.dopamine * 0.8 + target_dopamine * 0.2 * self.dopamine_gain
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
            self.serotonin *= 0.95  # Decay if meaningless

        self.serotonin = np.clip(self.serotonin, 0.0, 1.0)

        # Update history
        self.prev_energy = world_energy
        self.energy_history.append(world_energy)
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)

    def get_hormones(self):
        return {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "cortisol": self.cortisol,
        }


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

        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.consistency_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _encode_action(self, action, batch_size, device):
        """
        Accepts:
          - action as Long indices: (B,) or (B,1)
          - action as float vector: (B, action_dim)
        Returns:
          - action_vec: (B, action_dim) float32
        """
        if action.dim() == 2 and action.shape[1] == self.action_dim and action.dtype != torch.long:
            return action.float()

        if action.dim() == 1:
            action = action.view(-1, 1)
        if action.dim() == 2 and action.shape[1] == 1:
            idx = action.long().view(-1)
            idx = torch.clamp(idx, 0, self.action_dim - 1)
            one_hot = torch.zeros(batch_size, self.action_dim, device=device, dtype=torch.float32)
            one_hot.scatter_(1, idx.view(-1, 1), 1.0)
            return one_hot

        raise ValueError(f"Unsupported action shape: {tuple(action.shape)} dtype={action.dtype}")

    def forward(self, state, action):
        """
        Returns:
          next_state: (B, state_dim)
          pred_energy: (B, 1)
          pred_consistency: (B, 1) in [0,1]
        """
        if state.dim() != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must be (B,{self.state_dim}), got {tuple(state.shape)}")

        batch_size = state.shape[0]
        device = state.device
        action_vec = self._encode_action(action, batch_size, device)

        x = torch.cat([state, action_vec], dim=-1)
        h = self.trunk(x)

        next_state = self.state_head(h)
        pred_energy = self.energy_head(h)
        pred_consistency = self.consistency_head(h)

        return next_state, pred_energy, pred_consistency

    def simulate(self, state, action_id, num_actions=4):
        """
        Planner helper:
          state: (1, state_dim)
          action_id: int
        Returns:
          next_state: (1, state_dim)
          energy_cost: float
          pred_consistency: float
        """
        if not torch.is_tensor(state):
            raise ValueError("state must be a torch.Tensor")
        if state.dim() != 2 or state.shape[0] != 1 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must be (1,{self.state_dim}), got {tuple(state.shape)}")

        action_vec = torch.zeros(1, self.action_dim, device=state.device, dtype=torch.float32)
        action_index = int(action_id)
        action_index = max(0, min(action_index, min(num_actions, self.action_dim) - 1))
        action_vec[0, action_index] = 1.0

        with torch.no_grad():
            next_state, pred_energy, pred_consistency = self.forward(state, action_vec)

        return next_state, float(pred_energy.item()), float(pred_consistency.item())

    def loss(
        self,
        pred_next_state,
        pred_energy,
        pred_consistency,
        true_next_state,
        true_energy,
        true_consistency,
        w_state: float = 1.0,
        w_energy: float = 1.0,
        w_consistency: float = 1.0,
    ):
        """
        Compute combined JEPA losses.
        """
        if true_energy.dim() == 1:
            true_energy = true_energy.view(-1, 1)
        if true_consistency.dim() == 1:
            true_consistency = true_consistency.view(-1, 1)

        state_loss = torch.mean((pred_next_state - true_next_state) ** 2)
        energy_loss = torch.mean((pred_energy - true_energy) ** 2)
        consistency_loss = torch.mean((pred_consistency - true_consistency) ** 2)

        total = w_state * state_loss + w_energy * energy_loss + w_consistency * consistency_loss
        return total, {
            "state_loss": state_loss.detach(),
            "energy_loss": energy_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
        }

    def train_step(
        self,
        optimizer,
        batch,
        device=None,
        w_state: float = 1.0,
        w_energy: float = 1.0,
        w_consistency: float = 1.0,
        grad_clip: float | None = 1.0,
    ):
        """
        Run a single gradient update on a batch of transitions.
        """
        self.train()
        state = batch["state"]
        action = batch["action"]
        next_state = batch["next_state"]
        energy = batch["energy"]
        consistency = batch["consistency"]

        if device is not None:
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)
            energy = energy.to(device)
            consistency = consistency.to(device)

        pred_next_state, pred_energy, pred_consistency = self.forward(state, action)
        total_loss, parts = self.loss(
            pred_next_state,
            pred_energy,
            pred_consistency,
            next_state,
            energy,
            consistency,
            w_state=w_state,
            w_energy=w_energy,
            w_consistency=w_consistency,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()

        parts["total_loss"] = total_loss.detach()
        return parts
