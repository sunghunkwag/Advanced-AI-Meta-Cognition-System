import torch
from energy import JEPA_Predictor


def test_forward_accepts_action_indices_and_vectors():
    torch.manual_seed(0)
    model = JEPA_Predictor(state_dim=4, action_dim=3)
    state = torch.randn(2, 4)
    action_idx = torch.tensor([0, 2])
    next_state, pred_energy, pred_consistency = model(state, action_idx)

    assert next_state.shape == (2, 4)
    assert pred_energy.shape == (2, 1)
    assert pred_consistency.shape == (2, 1)

    action_vec = torch.eye(3)[:2]
    next_state_vec, energy_vec, cons_vec = model(state, action_vec)
    assert torch.allclose(pred_energy, energy_vec) is False or torch.allclose(pred_consistency, cons_vec) is False
    assert next_state_vec.shape == (2, 4)


def test_simulate_uses_predicted_energy():
    torch.manual_seed(1)
    model = JEPA_Predictor(state_dim=4, action_dim=3)
    state = torch.randn(1, 4)

    _, energy_a, _ = model.simulate(state, action_id=0, num_actions=3)
    _, energy_b, _ = model.simulate(state, action_id=1, num_actions=3)
    assert abs(energy_a - energy_b) > 1e-4


def test_train_step_reduces_loss_on_toy_batch():
    torch.manual_seed(42)
    model = JEPA_Predictor(state_dim=2, action_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    states = torch.randn(32, 2)
    actions = torch.randint(0, 2, (32, 1))
    next_states = states + actions.float()
    energies = torch.sum(next_states, dim=1, keepdim=True)
    consistencies = torch.sigmoid(energies)

    batch = {
        "state": states,
        "action": actions,
        "next_state": next_states,
        "energy": energies,
        "consistency": consistencies,
    }

    losses = []
    for _ in range(200):
        parts = model.train_step(optimizer, batch, grad_clip=None)
        losses.append(parts["total_loss"].item())

    assert losses[-1] < losses[0] * 0.5
