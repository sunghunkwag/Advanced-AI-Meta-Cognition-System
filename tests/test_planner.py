import torch
from planner import System2Planner


class DummyWorldModel:
    def __init__(self, scores):
        self.scores = scores

    def simulate(self, state, action):
        idx = int(torch.argmax(action))
        score = self.scores[idx]
        return state, torch.tensor([[score]]), torch.tensor([[1 - score]])


def test_planner_ranks_actions():
    scores = [0.3, 0.1, 0.5]
    world_model = DummyWorldModel(scores)
    planner = System2Planner(world_model, action_encoder=lambda x: x, depth=1, candidates=3)
    state = torch.zeros(1, 2)
    candidates = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])]
    result = planner.rollout(state, candidates, cortisol=0.0)
    assert result.action.tolist() == [0.0, 1.0, 0.0]
