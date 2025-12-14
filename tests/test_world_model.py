import torch
from world_model import WorldModel


def test_world_model_forward_shape_and_determinism():
    model = WorldModel(grid_size=4, action_dim=5)
    torch.manual_seed(0)
    state = torch.rand(2, 4, 4)
    action = torch.rand(2, 5)
    out1 = model(state, action)
    torch.manual_seed(0)
    out2 = model(state, action)
    for a, b in zip(out1, out2):
        assert torch.allclose(a, b)
    state_pred, energy_pred, consistency_pred = out1
    assert state_pred.shape == (2, 4, 4)
    assert energy_pred.shape == (2, 1)
    assert consistency_pred.shape == (2, 1)
