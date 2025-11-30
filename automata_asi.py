import torch
import torch.nn as nn
from automata import IntrinsicAutomata
from manifold import GraphAttentionManifold

class ManifoldAutomata(nn.Module):
    """
    The Brain + Memory.
    Combines Reasoning (Manifold) with Knowledge Preservation (Automata).
    """
    def __init__(self, state_dim=32, num_heads=4, v_truth=None):
        super(ManifoldAutomata, self).__init__()
        # Underlying GNN
        # Input features: 4 (from Vision)
        # Hidden: state_dim
        # Output: state_dim
        self.manifold = GraphAttentionManifold(nfeat=4, nhid=state_dim, nclass=state_dim, nheads=num_heads, truth_vector=v_truth)

        # Memory Manager
        self.automata = IntrinsicAutomata(self.manifold)

    def forward(self, x, adj, steps=1):
        # x: (B, N, F)
        # adj: (B, N, N)
        # GraphAttentionManifold expects (N, F) and (N, N) unbatched usually, or we need to adapt it.
        # Current GraphAttentionManifold implementation does NOT support batching explicitly in `forward` (uses mm).
        # But main_asi passes batched input (B=1).

        if x.dim() == 3:
             # Process single batch item
             x_in = x[0]
             adj_in = adj[0]
        else:
             x_in = x
             adj_in = adj

        z = self.manifold(x_in, adj_in)

        # Return as (1, 1, D) to match expected "current_state" shape (B, Time?, Dim)
        return z.unsqueeze(0)

    def ewc_loss(self):
        return self.automata.ewc_loss(self.manifold)

    def register_ewc_task(self, data, loss_fn):
        self.automata.crystallize(data)
