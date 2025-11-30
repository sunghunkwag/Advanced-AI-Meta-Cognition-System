import torch
import torch.nn as nn
from manifold import GraphAttentionManifold

class IntrinsicAutomata:
    """
    The Soul.
    Manages the learning process and crystallization (EWC).
    """
    def __init__(self, model):
        self.model = model
        self.fisher_matrix = {}
        self.optpar = {}
        self.crystallized = False
        self.ewc_lambda = 0.4

    def is_crystallized(self):
        return self.crystallized

    def crystallize(self, dataset_iterator):
        """
        Freeze the current knowledge by computing the Fisher Information Matrix.
        This corresponds to 'Meditation' or 'Enlightenment'.
        """
        if self.crystallized:
            return
            
        print("✨ [SOUL] Crystallizing Knowledge (EWC)...")
        self.model.eval()
        
        # Initialize Fisher Matrix
        for name, param in self.model.named_parameters():
            self.optpar[name] = param.data.clone()
            self.fisher_matrix[name] = torch.zeros_like(param.data)

        # Compute Fisher (Simplified: Gradients squared)
        # In a real scenario, we'd iterate over some memory buffer
        # Here we assume the current state is the 'truth'
        self.model.zero_grad()
        
        # Dummy forward pass to get gradients (Conceptual)
        # In practice, we need real data samples here.
        # For this simulation, we'll just lock the weights as they are.
        
        self.crystallized = True
        print("💎 [SOUL] Crystallization Complete. Weights are now resistant to change.")

    def ewc_loss(self, model):
        """
        Calculate the EWC loss penalty to preserve old memories.
        """
        if not self.crystallized:
            return 0
        
        loss = 0
        for name, param in model.named_parameters():
            fisher = self.fisher_matrix.get(name)
            optpar = self.optpar.get(name)
            if fisher is not None:
                # Standard EWC penalty: sum(F * (theta - theta*)^2)
                # Since we didn't compute real Fisher, we use identity (1.0) for strong locking
                loss += (self.ewc_lambda / 2) * torch.sum((param - optpar) ** 2)
        return loss

    def update_state(self, hormone_state):
        """
        Decide whether to crystallize based on the Heart's state.
        """
        dopamine, serotonin = hormone_state
        
        # If Serotonin (Peace) is very high, crystallize
        if serotonin > 0.9 and not self.crystallized:
            self.crystallize(None)


class ManifoldAutomata(nn.Module):
    """
    The Brain + Memory.
    Combines Reasoning (Manifold) with Knowledge Preservation (Automata).
    """
    def __init__(self, state_dim=32, num_heads=4, v_truth=None):
        super(ManifoldAutomata, self).__init__()
        # Underlying GNN
        # Input features: state_dim (because main_asi pads it)
        # Hidden: state_dim
        # Output: state_dim
        self.manifold = GraphAttentionManifold(nfeat=state_dim, nhid=state_dim, nclass=state_dim, nheads=num_heads, truth_vector=v_truth)

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
