"""Configuration management for Advanced AI Meta-Cognition System.

Centralized configuration for hyperparameters, architecture settings,
and experimental parameters.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class WorldConfig:
    """Configuration for the world/environment."""
    size: int = 16
    target_density: float = 0.1
    symmetry_weight: float = 1.0
    boredom_penalty_weight: float = 0.5


@dataclass
class ArchitectureConfig:
    """Neural architecture configuration."""
    latent_dim: int = 8
    gat_hidden_dim: int = 16
    gat_nfeat: int = 3
    action_decoder_hidden: int = 32
    jepa_hidden_dim: int = 64
    

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 0.01
    max_steps: int = 200
    curriculum_draw_steps: int = 10
    curriculum_mixed_steps: int = 20
    curriculum_draw_prob: float = 0.5
    
    # Loss weights
    energy_weight: float = 1.0
    consistency_weight: float = 1.0
    ewc_weight: float = 1.0
    

@dataclass
class NeuroChemicalConfig:
    """Neuro-chemical engine parameters."""
    initial_dopamine: float = 0.5
    initial_serotonin: float = 0.5
    dopamine_boost_factor: float = 2.0
    dopamine_decay: float = 0.95
    serotonin_boost: float = 0.1
    serotonin_decay: float = 0.98
    
    # Thresholds
    stability_variance_threshold: float = 0.01
    stability_history_window: int = 5
    consistency_threshold: float = 0.8
    energy_threshold: float = 0.2
    

@dataclass
class CrystallizationConfig:
    """Soul/EWC crystallization parameters."""
    serotonin_threshold: float = 0.95
    dopamine_threshold: float = 0.1
    consecutive_steps_required: int = 5
    ewc_lambda: float = 1000.0
    

@dataclass
class SystemConfig:
    """Complete system configuration."""
    world: WorldConfig = WorldConfig()
    architecture: ArchitectureConfig = ArchitectureConfig()
    training: TrainingConfig = TrainingConfig()
    neurochemical: NeuroChemicalConfig = NeuroChemicalConfig()
    crystallization: CrystallizationConfig = CrystallizationConfig()
    
    # Logging
    log_interval: int = 1
    verbose: bool = True
    save_checkpoints: bool = False
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 42


def get_default_config() -> SystemConfig:
    """Return default system configuration."""
    return SystemConfig()


def get_experimental_config() -> SystemConfig:
    """Return experimental configuration with enhanced parameters."""
    config = SystemConfig()
    
    # Enhanced architecture
    config.architecture.latent_dim = 16
    config.architecture.gat_hidden_dim = 32
    config.architecture.jepa_hidden_dim = 128
    
    # Extended training
    config.training.max_steps = 500
    config.training.learning_rate = 0.005
    
    # More sensitive neuro-chemical dynamics
    config.neurochemical.dopamine_boost_factor = 3.0
    config.neurochemical.serotonin_boost = 0.15
    
    return config
