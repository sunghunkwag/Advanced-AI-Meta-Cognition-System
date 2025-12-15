import torch

from soul import compute_truth_vector


def test_truth_vector_changes_with_symmetry():
    grid_symmetric = torch.zeros((6, 6))
    grid_symmetric[:, 2:4] = 1.0
    grid_symmetric = (grid_symmetric + torch.flip(grid_symmetric, dims=[1])) / 2

    grid_asymmetric = torch.zeros((6, 6))
    grid_asymmetric[:, :3] = 1.0

    truth_sym = compute_truth_vector(grid_symmetric)
    truth_asym = compute_truth_vector(grid_asymmetric)

    assert truth_sym.shape[0] == 32
    assert truth_asym.shape[0] == 32
    assert truth_sym[:8].mean() > truth_asym[:8].mean()
    assert not torch.allclose(truth_sym, truth_asym)
