from dataclasses import dataclass
from typing import Dict
import torch


@dataclass
class MetaState:
    lr: float
    epsilon: float
    ewc_lambda: float
    dopamine_gain: float


class MetaLearner:
    def __init__(self, config):
        self.config = config
        self.history = {"lr": [], "epsilon": [], "ewc": [], "dopamine_gain": []}

    def update(self, step: int, metrics: Dict[str, float], optimizer, exploration_ref: Dict[str, float], ewc_lambda_ref: Dict[str, float], heart):
        if not self.config.enabled:
            return None
        if step % self.config.update_interval != 0:
            return None

        energy_trend = metrics.get("energy_trend", 0.0)
        consistency_vol = metrics.get("consistency_vol", 0.0)
        action_div = metrics.get("action_diversity", 1.0)

        lr = optimizer.param_groups[0]["lr"]
        lr_factor = 1.0 - 0.1 * energy_trend
        new_lr = float(torch.clamp(torch.tensor(lr * lr_factor), self.config.lr_min, self.config.lr_max))
        for group in optimizer.param_groups:
            group["lr"] = new_lr

        epsilon = exploration_ref.get("epsilon", 0.1)
        eps_factor = 1.0 + 0.1 * consistency_vol
        new_eps = float(torch.clamp(torch.tensor(epsilon * eps_factor), self.config.epsilon_min, self.config.epsilon_max))
        exploration_ref["epsilon"] = new_eps

        ewc_lambda = ewc_lambda_ref.get("lambda", 0.1)
        ewc_factor = 1.0 + 0.1 * (1.0 / max(action_div, 1e-3))
        new_ewc = float(torch.clamp(torch.tensor(ewc_lambda * ewc_factor), self.config.ewc_min, self.config.ewc_max))
        ewc_lambda_ref["lambda"] = new_ewc

        dopamine_gain = getattr(heart, "dopamine_gain", 1.0)
        gain_factor = 1.0 + 0.05 * (metrics.get("prediction_error", 0.0))
        new_gain = float(torch.clamp(torch.tensor(dopamine_gain * gain_factor), self.config.dopamine_gain_min, self.config.dopamine_gain_max))
        heart.dopamine_gain = new_gain

        meta_state = MetaState(lr=new_lr, epsilon=new_eps, ewc_lambda=new_ewc, dopamine_gain=new_gain)
        self.history["lr"].append(new_lr)
        self.history["epsilon"].append(new_eps)
        self.history["ewc"].append(new_ewc)
        self.history["dopamine_gain"].append(new_gain)
        return meta_state
