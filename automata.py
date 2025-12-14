import torch
import torch.nn as nn

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

    def crystallize(self, nodes, adj):
        """
        Freeze the current knowledge by computing the Fisher Information Matrix.
        This corresponds to 'Meditation' or 'Enlightenment'.

        Args:
            nodes: Current graph nodes (from vision system)
            adj: Current adjacency matrix
        """
        if self.crystallized:
            return

        print("[SOUL] Crystallizing Knowledge (EWC)...")
        self.model.eval()

        for name, param in self.model.named_parameters():
            self.optpar[name] = param.data.clone()
            self.fisher_matrix[name] = torch.zeros_like(param.data)

        self.model.zero_grad()

        z = self.model(nodes, adj)
        consistency = self.model.check_consistency(z)

        consistency.backward()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.fisher_matrix[name] = param.grad.data.clone() ** 2
            else:
                self.fisher_matrix[name] = torch.zeros_like(param.data)

        self.model.train()
        self.crystallized = True
        print("[SOUL] Crystallization Complete. Weights are now resistant to change.")

    def ewc_loss(self, model):
        """
        Calculate the EWC loss penalty to preserve old memories.
        """
        if not self.crystallized:
            return torch.tensor(0.0)
        
        loss = 0
        for name, param in model.named_parameters():
            fisher = self.fisher_matrix.get(name)
            optpar = self.optpar.get(name)
            if fisher is not None:
                # Standard EWC penalty: sum(F * (theta - theta*)^2)
                # Since we didn't compute real Fisher, we use identity (1.0) for strong locking
                loss += (self.ewc_lambda / 2) * torch.sum((param - optpar) ** 2)
        return loss

    def update_state(self, hormone_state, nodes, adj):
        """
        Decide whether to crystallize based on the Heart's state.
        """
        dopamine, serotonin = hormone_state

        # If Serotonin (Peace) is very high, crystallize
        if serotonin > 0.9 and not self.crystallized:
            self.crystallize(nodes, adj)
