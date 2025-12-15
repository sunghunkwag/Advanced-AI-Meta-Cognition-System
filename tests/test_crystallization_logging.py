import torch

from automata import IntrinsicAutomata


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, nodes, adj):
        return self.w * nodes

    def check_consistency(self, z):
        return z.mean()

    def named_parameters(self, recurse: bool = True):  # type: ignore[override]
        return super().named_parameters(recurse=recurse)


def test_crystallization_records_checkpoints():
    model = DummyModel()
    automata = IntrinsicAutomata(model)
    nodes = torch.ones(1, 1, requires_grad=True)
    adj = torch.ones(1, 1)
    energy_history = [1.0, 0.3]
    consistency_history = [0.9] * 25

    automata.update_state((0.1, 0.9), nodes, adj, energy_history=energy_history, consistency_history=consistency_history, step=5)

    assert automata.ewc_tasks
    assert any("Checkpoint" in log for log in automata.crystallization_log)
