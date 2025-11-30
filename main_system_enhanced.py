"""Enhanced main system with configuration management and improved logging.

Integrates config.py and logger.py for better experimental control.
"""

import torch
import torch.optim as optim
import numpy as np
import argparse
import os
from pathlib import Path

# Core modules
from vision import VisionSystem
from manifold import GraphAttentionManifold
from energy_improved import NeuroChemicalEngine, ImprovedJEPA
from action_decoder import ActionDecoder
from automata import IntrinsicAutomata
from world import World
from soul import get_soul_vectors

# New modules
from config import SystemConfig, get_default_config, get_experimental_config
from logger import SystemLogger, StepMetrics
import time


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AdvancedAISystem:
    """Integrated system with all components."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = SystemLogger(verbose=config.verbose)
        
        # Set seed if specified
        if config.seed is not None:
            set_seed(config.seed)
        
        # Initialize world & perception
        self.world = World(size=config.world.size)
        self.vision = VisionSystem()
        
        # Initialize mind with soul
        v_id, v_truth, v_rej = get_soul_vectors(dim=config.architecture.latent_dim)
        self.mind = GraphAttentionManifold(
            nfeat=config.architecture.gat_nfeat,
            nhid=config.architecture.gat_hidden_dim,
            nclass=config.architecture.latent_dim,
            truth_vector=v_truth
        )
        
        # Initialize body
        self.body = ActionDecoder(latent_dim=config.architecture.latent_dim)
        
        # Initialize heart & soul
        self.heart = NeuroChemicalEngine(config.neurochemical)
        self.soul = IntrinsicAutomata(self.mind)
        
        # Optional: Initialize JEPA for planning
        self.jepa = ImprovedJEPA(
            state_dim=config.architecture.latent_dim,
            action_dim=4,  # 4 discrete actions
            hidden_dim=config.architecture.jepa_hidden_dim
        )
        
        # Optimizers
        self.optimizer = optim.Adam(
            list(self.mind.parameters()) + list(self.body.parameters()),
            lr=config.training.learning_rate
        )
        
        self.jepa_optimizer = optim.Adam(
            self.jepa.parameters(),
            lr=config.training.learning_rate * 0.5
        )
        
        # Move to device
        device = torch.device(config.device)
        self.mind.to(device)
        self.body.to(device)
        self.jepa.to(device)
        
    def run_step(self, step: int) -> StepMetrics:
        """Execute a single step of the system.
        
        Returns:
            StepMetrics object with all relevant information
        """
        config = self.config
        
        # === PERCEPTION ===
        world_state = self.world.get_state()
        nodes, adj = self.vision.perceive(world_state)
        
        # === MIND (Reasoning) ===
        z = self.mind(nodes, adj)
        consistency = self.mind.check_consistency(z)
        
        # === CALCULATE ENERGY ===
        world_energy = self.world.calculate_energy()
        
        # === HEART (Emotions) ===
        self.heart.update(world_energy, consistency.item())
        dopamine, serotonin = self.heart.get_hormones()
        state_mode = self.heart.get_state()
        
        # === SOUL (Crystallization Check) ===
        self.soul.update_state((dopamine, serotonin))
        ewc_loss = self.soul.ewc_loss(self.mind)
        
        # === BODY (Action Selection) ===
        action_logits, params = self.body(z)
        
        # Curriculum learning
        if step <= config.training.curriculum_draw_steps:
            action_logits = torch.zeros_like(action_logits)
            action_logits[0, 0] = 10.0  # Force DRAW
        elif step <= config.training.curriculum_mixed_steps:
            if np.random.rand() < config.training.curriculum_draw_prob:
                action_logits = torch.zeros_like(action_logits)
                action_logits[0, 0] = 10.0
        
        action = self.body.decode_action(action_logits, params)
        
        # === ACT ON WORLD ===
        self.world.apply_action(action)
        
        # === LEARNING ===
        loss = (config.training.energy_weight * torch.tensor(world_energy, dtype=torch.float32) +
                config.training.consistency_weight * (1.0 - consistency) +
                config.training.ewc_weight * ewc_loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.mind.parameters()) + list(self.body.parameters()), 
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Optional: Train JEPA world model
        # (Would need trajectory buffer for proper training)
        
        # Create metrics
        metrics = StepMetrics(
            step=step,
            timestamp=time.time(),
            mode=state_mode,
            dopamine=dopamine,
            serotonin=serotonin,
            world_energy=world_energy,
            consistency=consistency.item(),
            loss=loss.item(),
            ewc_loss=ewc_loss.item(),
            action_type=action['type'],
            action_params=action,
            grid_sum=self.world.grid.sum(),
            grid_density=self.world.grid.sum() / (self.world.size ** 2),
            is_crystallized=self.soul.is_crystallized()
        )
        
        return metrics
    
    def run(self) -> SystemLogger:
        """Run the complete system lifecycle.
        
        Returns:
            Logger with complete metrics history
        """
        config = self.config
        
        # Log initialization
        config_str = f"Latent Dim: {config.architecture.latent_dim} | Max Steps: {config.training.max_steps}"
        self.logger.log_init(config_str)
        
        # Main loop
        for step in range(1, config.training.max_steps + 1):
            try:
                metrics = self.run_step(step)
                
                if step % config.log_interval == 0:
                    self.logger.log_step(metrics)
                
                # Check crystallization
                if metrics.is_crystallized:
                    self.logger.log_crystallization(step)
                    break
                    
            except Exception as e:
                print(f"\n[ERROR] Step {step} failed: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Log completion
        final_metrics = self.logger.metrics_history[-1]
        self.logger.log_completion(final_metrics)
        
        return self.logger


def main():
    parser = argparse.ArgumentParser(description="Advanced AI Meta-Cognition System")
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'experimental'],
                       help='Configuration preset to use')
    parser.add_argument('--steps', type=int, default=None,
                       help='Override max steps')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--save-metrics', type=str, default=None,
                       help='Path to save metrics JSON')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'experimental':
        config = get_experimental_config()
    else:
        config = get_default_config()
    
    # Apply overrides
    if args.steps is not None:
        config.training.max_steps = args.steps
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    
    # Create and run system
    system = AdvancedAISystem(config)
    logger = system.run()
    
    # Save metrics if requested
    if args.save_metrics:
        logger.save_metrics(args.save_metrics)
        print(f"\n[SAVED] Metrics saved to {args.save_metrics}")
    
    # Print summary
    summary = logger.get_summary_statistics()
    print("\n[SUMMARY]")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
