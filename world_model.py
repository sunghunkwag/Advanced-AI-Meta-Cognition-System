import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TransitionBuffer:
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.data = []

    def add(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, energy: float, consistency: float):
        self.data.append((state.detach().clone(), action.detach().clone(), next_state.detach().clone(), energy, consistency))
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, batch_size: int = 32):
        if len(self.data) < batch_size:
            return None
        indices = torch.randperm(len(self.data))[:batch_size]
        states, actions, next_states, energies, consistencies = zip(*[self.data[i] for i in indices])
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(next_states),
            torch.tensor(energies, dtype=torch.float32).unsqueeze(1),
            torch.tensor(consistencies, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.data)


class WorldModel(nn.Module):
    """Simple JEPA-like predictor for grid world."""

    def __init__(self, grid_size: int, action_dim: int):
        super().__init__()
        self.grid_size = grid_size
        input_dim = grid_size * grid_size + action_dim
        hidden = 256
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden, grid_size * grid_size)
        self.energy_head = nn.Linear(hidden, 1)
        self.consistency_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_state = state.view(state.size(0), -1)
        x = torch.cat([flat_state, action], dim=1)
        h = self.encoder(x)
        state_pred = torch.sigmoid(self.state_head(h)).view(-1, self.grid_size, self.grid_size)
        energy_pred = self.energy_head(h)
        consistency_pred = torch.sigmoid(self.consistency_head(h))
        return state_pred, energy_pred, consistency_pred

    @torch.no_grad()
    def simulate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(state, action)


class WorldModelTrainer:
    def __init__(self, model: WorldModel, lr: float = 1e-3):
        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = TransitionBuffer()
        self.loss_history = []

    def add_transition(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, energy: float, consistency: float):
        self.buffer.add(state, action, next_state, energy, consistency)

    def train_step(self, batch_size: int = 32):
        batch = self.buffer.sample(batch_size)
        if batch is None:
            return None
        states, actions, next_states, energies, consistencies = batch
        pred_state, pred_energy, pred_consistency = self.model(states, actions)
        loss_state = F.mse_loss(pred_state, next_states)
        loss_energy = F.mse_loss(pred_energy, energies)
        loss_consistency = F.mse_loss(pred_consistency, consistencies)
        loss = loss_state + loss_energy + loss_consistency
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.loss_history.append(loss.item())
        return {
            "total": loss.item(),
            "state": loss_state.item(),
            "energy": loss_energy.item(),
            "consistency": loss_consistency.item(),
        }
