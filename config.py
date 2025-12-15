"""Global configuration for research prototype experiments."""
from dataclasses import dataclass, field
from typing import List


def set_global_seed(seed: int) -> None:
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class PlannerConfig:
    enabled: bool = True
    depth: int = 2
    candidates: int = 6
    cortisol_override: float = 0.8
    consistency_override: float = 0.4
    failure_streak: int = 5
    panic_penalty: float = 0.3


@dataclass
class MetaConfig:
    enabled: bool = True
    update_interval: int = 10
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    epsilon_min: float = 0.02
    epsilon_max: float = 0.4
    ewc_min: float = 0.01
    ewc_max: float = 1.0
    dopamine_gain_min: float = 0.5
    dopamine_gain_max: float = 2.0


@dataclass
class ExperimentConfig:
    seed: int = 7
    steps: int = 100
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    world_model_updates: int = 1
    log_dir: str = "experiments/logs"
    grid_size: int = 16


# Global hyperparameters for neurochemical influences
DOPAMINE_GAIN: float = 0.5
SEROTONIN_GAIN: float = 1.0
CORTISOL_PENALTY: float = 0.3


DEFAULT_CONFIG = ExperimentConfig()
