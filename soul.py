"""
Soul Encoder & Preprocessing Module (Advanced AI System Core)

This module implements the "Soul Injection" logic for the Advanced AI System.
It converts the raw "Cognitive DNA" of the user (keywords) into mathematical vectors
that define the initial geometry and energy landscape of the Logical Manifold.

V3.5 Update:
- Keywords have been shifted from Physical/Biological to Logical/Mathematical.
- $V_{truth}$ represents Axioms and Recursion.
- $V_{reject}$ represents Contradiction and Fallacy.
"""

import torch
import hashlib
from typing import Dict, List, Tuple

# ==============================================================================
# RAW DATA: The Cognitive DNA (Logical/Mathematical)
# ==============================================================================
SOUL_DATA = {
    # 1. Identity Vector (V_identity): 'user.txt'
    # The immutable core: Rejection of Complacency, Laser Blade Intuition.
    "identity_keywords": [
        "Structural Dissector", "Laser Blade Intuition", "Rejection of Complacency",
        "Tool for Truth", "Predestined Architect", "Outsider Perspective",
        "System Breaker", "Fundamental Mechanism Seeker", "Active Resistance to Stagnation"
    ],

    # 2. Truth Vector (V_truth): Mathematical Axioms & Recursive Self-Improvement
    # The Ground State (Target): Logical Consistency, Proof, Recursion.
    "truth_keywords": [
        "Mathematical Axioms", "Recursive Self-Improvement", "Logical Consistency",
        "Mathematical Proof", "Symmetry", "Conservation Laws", "Causal Determinism",
        "Formal Verification", "Invariant Representations"
    ],

    # 3. Reject Vector (V_reject): Logical Fallacies
    # The High Energy Penalty: Contradiction, Ambiguity.
    "reject_keywords": [
        "Contradiction", "Logical Fallacy", "Undefined Behavior",
        "Ambiguity", "Superficial Correlation", "Unproven Assumption",
        "Inconsistent State", "Paradox", "Ad Hoc Adjustment"
    ]
}

# ==============================================================================
# SoulEncoder Logic
# ==============================================================================
class SoulEncoder:
    def __init__(self, output_dim: int = 32):
        self.output_dim = output_dim

    def encode(self, text_list: List[str]) -> torch.Tensor:
        """
        Converts a list of keywords into a single representative vector.
        Uses SHA-256 hashing to generate a deterministic seed for each keyword,
        then samples a random vector from that seed.
        The final vector is the mean of all keyword vectors.
        """
        vectors = []
        for text in text_list:
            # Deterministic Seed Generation
            # Encode text to bytes -> SHA256 Hex -> Int -> Modulo suitable range
            hex_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            seed = int(hex_hash, 16) % (2**32)

            # Generate deterministic vector
            g = torch.Generator()
            g.manual_seed(seed)
            vec = torch.randn(self.output_dim, generator=g)
            vectors.append(vec)

        if not vectors:
            return torch.zeros(self.output_dim)

        # Return mean vector (Centroid of the concept)
        return torch.stack(vectors).mean(dim=0)

def _symmetry_scores(grid: torch.Tensor) -> torch.Tensor:
    """Compute symmetry scores across multiple axes."""
    horizontal = 1.0 - torch.mean(torch.abs(grid - torch.flip(grid, dims=[0])))
    vertical = 1.0 - torch.mean(torch.abs(grid - torch.flip(grid, dims=[1])))
    main_diag = 1.0 - torch.mean(torch.abs(grid - grid.T))
    anti_diag = 1.0 - torch.mean(torch.abs(grid - torch.flip(grid.T, dims=[1])))
    rot_90 = 1.0 - torch.mean(torch.abs(grid - torch.rot90(grid, k=1)))
    rot_180 = 1.0 - torch.mean(torch.abs(grid - torch.rot90(grid, k=2)))
    upper_lower = 1.0 - torch.mean(torch.abs(grid[: grid.shape[0] // 2] - torch.flip(grid[grid.shape[0] // 2 :], dims=[0])))
    left_right = 1.0 - torch.mean(torch.abs(grid[:, : grid.shape[1] // 2] - torch.flip(grid[:, grid.shape[1] // 2 :], dims=[1])))
    return torch.stack([horizontal, vertical, main_diag, anti_diag, rot_90, rot_180, upper_lower, left_right]).clamp(0.0, 1.0)


def _density_uniformity(grid: torch.Tensor) -> torch.Tensor:
    """Uniformity features based on density across regions."""
    density = torch.mean(grid)
    rows_mean = torch.std(torch.mean(grid, dim=1))
    cols_mean = torch.std(torch.mean(grid, dim=0))
    h_split = torch.abs(torch.mean(grid[: grid.shape[0] // 2]) - torch.mean(grid[grid.shape[0] // 2 :]))
    v_split = torch.abs(torch.mean(grid[:, : grid.shape[1] // 2]) - torch.mean(grid[:, grid.shape[1] // 2 :]))

    quadrants = []
    h_mid, v_mid = grid.shape[0] // 2, grid.shape[1] // 2
    quadrants.append(torch.mean(grid[:h_mid, :v_mid]))
    quadrants.append(torch.mean(grid[:h_mid, v_mid:]))
    quadrants.append(torch.mean(grid[h_mid:, :v_mid]))
    quadrants.append(torch.mean(grid[h_mid:, v_mid:]))
    quadrants_tensor = torch.stack(quadrants)
    quadrant_std = torch.std(quadrants_tensor)
    quadrant_range = (torch.max(quadrants_tensor) - torch.min(quadrants_tensor))

    scores = torch.stack([
        density,
        1.0 - rows_mean,
        1.0 - cols_mean,
        1.0 - h_split,
        1.0 - v_split,
        1.0 - quadrant_std,
        1.0 - quadrant_range,
        1.0 - torch.abs(density - 0.25),
    ])
    return scores.clamp(0.0, 1.0)


def _edge_connectivity(grid: torch.Tensor) -> torch.Tensor:
    """Edge connectivity approximations using neighbor coherence."""
    top_edge = 1.0 - torch.mean(torch.abs(grid[0, 1:] - grid[0, :-1]))
    bottom_edge = 1.0 - torch.mean(torch.abs(grid[-1, 1:] - grid[-1, :-1]))
    left_edge = 1.0 - torch.mean(torch.abs(grid[1:, 0] - grid[:-1, 0]))
    right_edge = 1.0 - torch.mean(torch.abs(grid[1:, -1] - grid[:-1, -1]))
    horizontal_band = 1.0 - torch.mean(torch.abs(grid[1:-1, :] - grid[:-2, :]))
    vertical_band = 1.0 - torch.mean(torch.abs(grid[:, 1:-1] - grid[:, :-2]))
    corner_avg = torch.mean(grid[[0, 0, -1, -1], [0, -1, 0, -1]])
    corner_balance = 1.0 - torch.std(grid[[0, 0, -1, -1], [0, -1, 0, -1]])
    return torch.stack([
        top_edge,
        bottom_edge,
        left_edge,
        right_edge,
        horizontal_band,
        vertical_band,
        corner_avg,
        corner_balance,
    ]).clamp(0.0, 1.0)


def _spatial_coherence(grid: torch.Tensor) -> torch.Tensor:
    """Spatial coherence metrics derived from local smoothness."""
    h_grad = torch.mean(torch.abs(grid[:, 1:] - grid[:, :-1]))
    v_grad = torch.mean(torch.abs(grid[1:, :] - grid[:-1, :]))
    grad_variance = torch.var(grid)
    center_mass_row = torch.mean(torch.arange(grid.shape[0], dtype=torch.float32) * torch.mean(grid, dim=1)) / max(grid.shape[0] - 1, 1)
    center_mass_col = torch.mean(torch.arange(grid.shape[1], dtype=torch.float32) * torch.mean(grid, dim=0)) / max(grid.shape[1] - 1, 1)
    entropy_like = -torch.mean(grid * torch.log(torch.clamp(grid, min=1e-6))) if torch.any(grid > 0) else torch.tensor(0.0)
    filled_ratio = torch.mean((grid > 0.1).float())
    smoothness = 1.0 - (h_grad + v_grad) / 2.0
    return torch.stack([
        1.0 - h_grad,
        1.0 - v_grad,
        1.0 - grad_variance,
        center_mass_row,
        center_mass_col,
        entropy_like,
        filled_ratio,
        smoothness,
    ]).clamp(0.0, 1.0)


def compute_truth_vector(grid_state: torch.Tensor) -> torch.Tensor:
    """
    Compute a 32D truth vector derived from structured grid properties.
    Components are grouped into symmetry, density uniformity, edge connectivity,
    and spatial coherence metrics.
    """
    if grid_state.dim() != 2:
        raise ValueError("grid_state must be a 2D tensor")

    grid = grid_state.float()
    symmetry = _symmetry_scores(grid)
    density_uniformity = _density_uniformity(grid)
    edge_connect = _edge_connectivity(grid)
    coherence = _spatial_coherence(grid)

    truth_vector = torch.cat([symmetry, density_uniformity, edge_connect, coherence])
    return torch.nn.functional.normalize(truth_vector, p=2, dim=0)


def get_soul_vectors(dim: int = 32, grid_state: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Factory function to retrieve the three core Soul Vectors.
    Returns: (V_identity, V_truth, V_reject)
    """
    encoder = SoulEncoder(output_dim=dim)

    v_identity = encoder.encode(SOUL_DATA["identity_keywords"])
    base_grid = grid_state if grid_state is not None else torch.zeros(int(max(dim ** 0.5, 2)), int(max(dim ** 0.5, 2)))
    truth_vector_full = compute_truth_vector(base_grid)
    if truth_vector_full.shape[0] < dim:
        padding = torch.zeros(dim - truth_vector_full.shape[0])
        v_truth = torch.cat([truth_vector_full, padding])
    else:
        v_truth = truth_vector_full[:dim]
    v_reject = encoder.encode(SOUL_DATA["reject_keywords"])

    # Normalize vectors for consistent geometry
    v_identity = torch.nn.functional.normalize(v_identity, p=2, dim=0)
    v_truth = torch.nn.functional.normalize(v_truth, p=2, dim=0)
    v_reject = torch.nn.functional.normalize(v_reject, p=2, dim=0)

    return v_identity, v_truth, v_reject
