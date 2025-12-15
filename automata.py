import torch
import torch.nn as nn

class IntrinsicAutomata:
    """
    The Soul.
    Manages the learning process and crystallization (EWC).

    This module now supports multiple crystallization checkpoints to better
    capture distinct learning milestones. Each checkpoint stores its own
    Fisher information estimate and locking strength so downstream tasks can
    selectively regularize toward the appropriate anchor state.
    """

    def __init__(self, model):
        self.model = model
        self.fisher_matrix = {}
        self.optpar = {}
        self.crystallized = False
        self.ewc_lambda = 0.4
        self.ewc_tasks = {}
        self.crystallization_log = []

    def is_crystallized(self):
        return self.crystallized

    def crystallize(self, nodes, adj, checkpoint_name="default", metadata=None):
        """
        Freeze the current knowledge by computing the Fisher Information Matrix.
        This corresponds to 'Meditation' or 'Enlightenment'.

        Args:
            nodes: Current graph nodes (from vision system)
            adj: Current adjacency matrix
            checkpoint_name: Identifier for the crystallization milestone
            metadata: Optional dict with contextual notes (step, reason, etc.)
        """
        print(f"[SOUL] Crystallizing Knowledge (EWC) at checkpoint '{checkpoint_name}'...")
        self.model.eval()

        fisher_matrix = {}
        optpar = {}

        for name, param in self.model.named_parameters():
            optpar[name] = param.data.clone()
            fisher_matrix[name] = torch.zeros_like(param.data)

        self.model.zero_grad()

        z = self.model(nodes, adj)
        consistency = self.model.check_consistency(z)

        consistency.backward()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                fisher_matrix[name] = param.grad.data.clone() ** 2
            else:
                fisher_matrix[name] = torch.zeros_like(param.data)

        self.model.train()
        self.crystallized = True
        self.ewc_tasks[checkpoint_name] = {
            "fisher": fisher_matrix,
            "optpar": optpar,
            "metadata": metadata or {},
        }
        log_message = (
            f"[EWC] Checkpoint {checkpoint_name} locked"
            f" with fisher_magnitude={sum(v.sum().item() for v in fisher_matrix.values()):.4f}"
        )
        print(log_message)
        self.crystallization_log.append(log_message)

    def ewc_loss(self, model):
        """
        Calculate the EWC loss penalty to preserve old memories.
        Multiple checkpoints are aggregated with equal weight unless a custom
        ``lambda`` is provided inside the stored metadata.
        """
        if not self.ewc_tasks:
            return torch.tensor(0.0)

        losses = []
        for checkpoint_name, payload in self.ewc_tasks.items():
            fisher = payload.get("fisher", {})
            optpar = payload.get("optpar", {})
            lambda_i = payload.get("metadata", {}).get("lambda", self.ewc_lambda)
            checkpoint_loss = 0.0
            for name, param in model.named_parameters():
                fisher_matrix = fisher.get(name)
                anchor = optpar.get(name)
                if fisher_matrix is not None and anchor is not None:
                    checkpoint_loss += torch.sum(fisher_matrix * (param - anchor) ** 2)
            losses.append(0.5 * lambda_i * checkpoint_loss)

        return torch.stack([loss if torch.is_tensor(loss) else torch.tensor(loss) for loss in losses]).sum()

    def update_state(
        self,
        hormone_state,
        nodes,
        adj,
        energy_history=None,
        consistency_history=None,
        step: int = 0,
    ):
        """
        Decide whether to crystallize based on the Heart's state.
        Multi-criteria triggers model the "checkpoint" concept:
        - Checkpoint A: Energy drops below half of the starting level
        - Checkpoint B: High consistency sustained for 20 steps
        - Checkpoint C: Stable dopamine/serotonin ratio
        """
        dopamine, serotonin = hormone_state
        energy_history = energy_history or []
        consistency_history = consistency_history or []

        if not energy_history:
            return

        initial_energy = energy_history[0]
        current_energy = energy_history[-1]
        energy_threshold_hit = current_energy < 0.5 * initial_energy

        consistency_streak = (
            len(consistency_history) >= 20
            and all(val > 0.8 for val in consistency_history[-20:])
        )

        ratio_history = []
        if len(consistency_history) >= 50:
            # Proxy ratio stability using last 50 serotonin values normalized by dopamine
            ratio_history = [
                (consistency_history[i] if abs(dopamine) < 1e-3 else serotonin / (dopamine + 1e-3))
                for i in range(-50, 0)
            ]
        ratio_stable = bool(ratio_history) and (torch.tensor(ratio_history).std().item() < 0.1)

        if energy_threshold_hit and "A" not in self.ewc_tasks:
            self.crystallize(nodes, adj, checkpoint_name="A", metadata={"step": step})
        if consistency_streak and "B" not in self.ewc_tasks:
            self.crystallize(nodes, adj, checkpoint_name="B", metadata={"step": step})
        if ratio_stable and "C" not in self.ewc_tasks:
            self.crystallize(nodes, adj, checkpoint_name="C", metadata={"step": step})
