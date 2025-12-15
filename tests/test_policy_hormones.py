import torch
import torch.nn.functional as F

from energy import NeuroChemicalEngine


def test_dopamine_spike_increases_policy_gradient():
    engine = NeuroChemicalEngine()

    engine.prev_energy = 1.0
    high_reward = engine.update(
        world_energy=0.2,
        consistency_score=0.8,
        density=0.3,
        symmetry=0.7,
        prediction_error=0.2,
    )

    engine.prev_energy = 1.0
    low_reward = engine.update(
        world_energy=0.9,
        consistency_score=0.2,
        density=0.1,
        symmetry=0.1,
        prediction_error=0.9,
    )

    logits = torch.zeros((1, 2), requires_grad=True)
    log_probs = F.log_softmax(logits, dim=-1)

    loss_high = -log_probs[0, 1] * torch.tensor(high_reward)
    loss_high.backward()
    grad_high = logits.grad.detach().clone()

    logits.grad.zero_()
    loss_low = -F.log_softmax(logits, dim=-1)[0, 1] * torch.tensor(low_reward)
    loss_low.backward()
    grad_low = logits.grad.detach().clone()

    assert grad_high[0, 1].abs() > grad_low[0, 1].abs()
