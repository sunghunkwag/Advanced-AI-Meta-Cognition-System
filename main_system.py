import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from collections import defaultdict

# Import Core Modules (Original)
from vision import VisionSystem
from manifold import GraphAttentionManifold
from energy import NeuroChemicalEngine
from action_decoder import ActionDecoder
from automata import IntrinsicAutomata
from world import World
from soul import get_soul_vectors

# Import AGI Extensions
try:
    from memory import MemorySystem, ValueHierarchy, SelfModel, NeuroAgent, LifelongRunner
    AGI_EXTENSIONS_AVAILABLE = True
except ImportError:
    print("[WARNING] AGI extensions not available. Running original system.")
    AGI_EXTENSIONS_AVAILABLE = False


class AGISystem:
    """Unified AGI Architecture preserving original neuro-chemical core"""
    
    def __init__(self, use_agi_extensions=True):
        self.use_agi_extensions = use_agi_extensions and AGI_EXTENSIONS_AVAILABLE
        
        # Original Core Components
        self.world = World(size=16)
        self.vision = VisionSystem()
        LATENT_DIM = 8
        
        v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
        self.mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=v_truth)
        self.body = ActionDecoder(latent_dim=LATENT_DIM)
        self.heart = NeuroChemicalEngine()
        self.soul = IntrinsicAutomata(self.mind)
        
        # Original Optimizer
        self.optimizer = optim.Adam(
            list(self.mind.parameters()) + list(self.body.parameters()),
            lr=0.01
        )
        
        # AGI Extension Systems (if available)
        if self.use_agi_extensions:
            self.memory = MemorySystem()
            self.memory.load_semantic_memory()
            self.values = ValueHierarchy()
            self.values.load_state()
            self.self_model = SelfModel(self.memory)
            self.lifelong_runner = LifelongRunner()
        else:
            self.memory = None
            self.values = None
            self.self_model = None
            self.lifelong_runner = None
        
        # Tracking
        self.energy_history = []
        self.action_history = []
        self.best_energy = float('inf')
        self.step_count = 0
    
    def run_episode(self, max_steps=1000, environment=None):
        """Run a single episode with optional AGI extensions"""
        if environment:
            self.world = environment
        
        episode_data = {
            'actions': [],
            'energies': [],
            'hormones': [],
            'gat_patterns': {},
            'final_energy': 0.0
        }
        
        # Get hyperparameters from self-model if available
        if self.use_agi_extensions:
            hypers = self.self_model.get_hyperparameters()
            epsilon = hypers['epsilon']
            current_truth = self.values.compute_current_truth()
            self.mind.update_truth_vector(current_truth)
        else:
            epsilon = 0.1  # Default
        
        step = 0
        while step < max_steps:
            step += 1
            self.step_count += 1
            
            # === PERCEPTION ===
            world_state = self.world.get_state()
            nodes, adj = self.vision.perceive(world_state)
            
            # === MIND (Reasoning with optional memory-enhanced truth) ===
            if self.use_agi_extensions and self.memory:
                memory_truth = self.memory.get_truth_modification(current_truth)
                self.mind.update_truth_vector(memory_truth)
            
            z = self.mind(nodes, adj)
            consistency = self.mind.check_consistency(z)
            
            # Extract patterns for semantic memory
            if self.use_agi_extensions:
                patterns = self.mind.extract_patterns()
                episode_data['gat_patterns'].update(patterns)
            
            # === CALCULATE ENERGY ===
            world_energy = self.world.calculate_energy()
            episode_data['energies'].append(world_energy)
            
            # === HEART (Emotions) ===
            self.heart.update(world_energy, consistency.item())
            dopamine, serotonin = self.heart.get_hormones()
            state_mode = self.heart.get_state()
            episode_data['hormones'].append((dopamine, serotonin))
            
            # === SOUL (Crystallization Check) ===
            self.soul.update_state((dopamine, serotonin))
            ewc_loss = self.soul.ewc_loss(self.mind)
            
            # === BODY (Action Selection) ===
            action_logits, params = self.body(z)
            
            # Curriculum Learning (Original)
            if step <= 10:
                action_logits = torch.zeros_like(action_logits)
                action_logits[0, 0] = 10.0  # Force DRAW
            elif step <= 20:
                if np.random.rand() < 0.5:
                    action_logits = torch.zeros_like(action_logits)
                    action_logits[0, 0] = 10.0
            else:
                # Epsilon-Greedy Exploration
                if np.random.rand() < epsilon:
                    action_logits = torch.randn_like(action_logits)
            
            action = self.body.decode_action(action_logits, params)
            action_type = action['type']
            episode_data['actions'].append(action_type)
            self.action_history.append(action_type)
            
            # === ACT ON WORLD ===
            self.world.apply_action(action)
            
            # === LEARNING (Backpropagation) ===
            loss = torch.tensor(world_energy, dtype=torch.float32)
            loss = loss + (1.0 - consistency)
            loss = loss + ewc_loss
            
            # === REWARD SHAPING (Original Logic Preserved) ===
            self.energy_history.append(world_energy)
            if len(self.energy_history) > 10:
                recent_avg = np.mean(self.energy_history[-10:])
                prev_avg = np.mean(self.energy_history[-20:-10]) if len(self.energy_history) >= 20 else recent_avg
                improvement = prev_avg - recent_avg
                if improvement > 0:
                    loss = loss - torch.tensor(improvement * 0.1, dtype=torch.float32)
            
            # Action Diversity Bonus (Original)
            if len(self.action_history) >= 20:
                recent_actions = self.action_history[-20:]
                unique_actions = len(set(recent_actions))
                diversity_bonus = (unique_actions - 2) * 0.02
                loss = loss - torch.tensor(diversity_bonus, dtype=torch.float32)
            
            # === ADAPTIVE LEARNING RATE (Original) ===
            lr = 0.01 * (0.95 ** (step // 50))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.mind.parameters()) + list(self.body.parameters()), max_norm=1.0)
            self.optimizer.step()
            
            # Track best energy
            if world_energy < self.best_energy:
                self.best_energy = world_energy
            
            # === LOGGING ===
            if step % 50 == 1 or step <= 50:
                epsilon_val = epsilon if self.use_agi_extensions else 0.1
                print(f"Step {step:03d} | {state_mode:5s} | "
                      f"D:{dopamine:.2f} S:{serotonin:.2f} | "
                      f"E:{world_energy:.4f} | "
                      f"L:{loss.item():.4f} | "
                      f"LR:{lr:.5f} | "
                      f"ε:{epsilon_val:.3f} | "
                      f"{action_type:10s} | "
                      f"Grid:{self.world.grid.sum():.1f}")
            
            # === CHECK CRYSTALLIZATION ===
            if self.soul.is_crystallized():
                print(f"\n{"="*60}")
                print(f"[NIRVANA] Mind crystallized at step {step}")
                print(f"{"="*60}")
                break
        
        episode_data['final_energy'] = world_energy
        
        # === AGI EXTENSION INTEGRATION ===
        if self.use_agi_extensions:
            # Store to memory
            self.memory.store_episode(episode_data)
            
            # Update self-model and values
            self.self_model.analyze_performance()
            performance_metrics = self.self_model.get_performance_metrics()
            self.values.update_weights(performance_metrics)
            
            # Log AGI metrics
            profile = self.self_model.current_profile
            semantic_size = len(self.memory.semantic_memory)
            print(f"  [AGI] Profile: {profile} | Semantic Rules: {semantic_size}")
        
        return world_energy, step
    
    def run_lifelong(self, max_episodes=10):
        """Run lifelong learning if extensions available"""
        if not self.use_agi_extensions:
            print("[LIFELONG] Extensions required for lifelong learning")
            return 0, 0
        
        print(f"[LIFELONG] Starting AGI-enhanced lifetime")
        total_episodes = 0
        total_steps = 0
        
        for episode in range(max_episodes):
            print(f"\n[EPISODE {episode+1:02d}] Profile: {self.self_model.current_profile}")
            energy, steps = self.run_episode()
            total_episodes += 1
            total_steps += steps
            
            print(f"  Energy: {energy:.4f} | Steps: {steps} | "
                  f"Truth Norm: {self.values.compute_current_truth().norm():.3f}")
            
            # Periodic checkpoint
            if (episode + 1) % 5 == 0:
                self.save_checkpoint(episode + 1)
        
        self.save_checkpoint(total_episodes)
        return total_episodes, total_steps
    
    def save_checkpoint(self, episode_num):
        """Save system state"""
        if not self.use_agi_extensions:
            return
        
        checkpoint_dir = './checkpoints/'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'episode': episode_num,
            'total_steps': self.step_count,
            'best_energy': self.best_energy,
            'memory_size': len(self.memory.episodic_memory),
            'semantic_rules': len(self.memory.semantic_memory),
            'current_profile': self.self_model.current_profile,
            'value_weights': {k: float(v) for k, v in self.values.weights.items()}
        }
        
        # Save systems
        self.memory.save_semantic_memory()  # Handled in store_episode
        self.values.save_state()
        
        with open(f'{checkpoint_dir}/agi_checkpoint_{episode_num:04d}.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save model weights
        torch.save({
            'mind_state_dict': self.mind.state_dict(),
            'body_state_dict': self.body.state_dict(),
            'soul_fisher': self.soul.fisher_information,
            'optimizer_state': self.optimizer.state_dict(),
            'episode': episode_num,
            'step_count': self.step_count
        }, f'{checkpoint_dir}/agi_model_weights_{episode_num:04d}.pt')
        
        print(f"[CHECKPOINT] Saved AGI state at episode {episode_num}")
    
    def load_checkpoint(self, episode_num):
        """Load AGI system from checkpoint"""
        if not self.use_agi_extensions:
            return False
        
        checkpoint_dir = './checkpoints/'
        try:
            # Load systems
            self.memory.load_semantic_memory()
            self.values.load_state()
            
            # Load model weights
            checkpoint_path = f'{checkpoint_dir}/agi_model_weights_{episode_num:04d}.pt'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.mind.load_state_dict(checkpoint['mind_state_dict'])
                self.body.load_state_dict(checkpoint['body_state_dict'])
                self.soul.fisher_information = checkpoint['soul_fisher']
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                
                self.step_count = checkpoint.get('step_count', 0)
                self.best_energy = checkpoint.get('best_energy', float('inf'))
                
                # Load meta checkpoint for counters
                meta_path = f'{checkpoint_dir}/agi_checkpoint_{episode_num:04d}.json'
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    self.step_count = meta.get('total_steps', self.step_count)
                    
                print(f"[CHECKPOINT] Loaded AGI state from episode {episode_num}, steps: {self.step_count}")
                return True
            else:
                print(f"[CHECKPOINT] Model weights not found for episode {episode_num}")
                return False
        except Exception as e:
            print(f"[CHECKPOINT] Load failed: {e}")
            return False


def test_agi_system():
    """Comprehensive AGI system test"""
    print("[TEST] Starting AGI Architecture Validation")
    print("="*60)
    
    # Test 1: Original System (No Extensions)
    print("\n[TEST 1] Original Neuro-Chemical System")
    original_system = AGISystem(use_agi_extensions=False)
    energy1, steps1 = original_system.run_episode(max_steps=200)
    print(f"  Original: Energy={energy1:.4f}, Steps={steps1}, Best={original_system.best_energy:.4f}")
    
    # Test 2: AGI Extensions (if available)
    if AGI_EXTENSIONS_AVAILABLE:
        print("\n[TEST 2] AGI Extensions System")
        agi_system = AGISystem(use_agi_extensions=True)
        energy2, steps2 = agi_system.run_episode(max_steps=200)
        
        # Test lifelong learning
        print("\n[TEST 3] Lifelong Learning (3 episodes)")
        episodes, total_steps = agi_system.run_lifelong(max_episodes=3)
        
        print(f"  AGI Single: Energy={energy2:.4f}, Steps={steps2}")
        print(f"  Lifelong: Episodes={episodes}, Total Steps={total_steps}")
        print(f"  Semantic Memory: {len(agi_system.memory.semantic_memory)} rules")
        print(f"  Current Profile: {agi_system.self_model.current_profile}")
        
        # Test checkpoint
        agi_system.save_checkpoint(1)
        load_success = agi_system.load_checkpoint(1)
        print(f"  Checkpoint Test: {'PASS' if load_success else 'FAIL'}")
        
        # Test multi-agent (simplified)
        print("\n[TEST 4] Multi-Agent Social Dynamics")
        from memory import MultiAgentWorld
        multi_world = MultiAgentWorld(num_agents=2)
        agent1 = agi_system.lifelong_runner.create_agent(1001)
        agent2 = agi_system.lifelong_runner.create_agent(1002)
        multi_world.add_agent(agent1)
        multi_world.add_agent(agent2)
        
        avg_energy, social_score = multi_world.run_social_episode(max_steps=50)
        print(f"  Multi-Agent: Energy={avg_energy:.4f}, Social Score={social_score:.4f}")
        
        # Final validation
        if (energy2 < 0.3 and episodes >= 3 and load_success and social_score > 0):
            print("\n[VERIFICATION] ✓ AGI Architecture Extensions PASSED")
            print("  - Core neuro-chemical system preserved")
            print("  - Memory, values, self-model functional")
            print("  - Lifelong learning operational")
            print("  - Multi-agent social dynamics working")
            return True
        else:
            print("\n[VERIFICATION] ✗ AGI Extensions Validation FAILED")
            return False
    else:
        print("\n[VERIFICATION] Original System Only - Extensions Not Available")
        return energy1 < 0.3
    
    return False


def main_agi():
    """Main entry point for AGI-enhanced system"""
    print("[INIT] Advanced AGI Meta-Cognition System")
    print("="*60)
    
    if not test_agi_system():
        print("[ERROR] AGI system validation failed. Running original system.")
        # Fallback to original
        from main_system import main as original_main
        original_main()
        return
    
    print("\n[AGI] All systems verified. Running full AGI simulation...")
    
    # Run full AGI lifelong simulation
    agi_system = AGISystem(use_agi_extensions=True)
    episodes, total_steps = agi_system.run_lifelong(max_episodes=10)
    
    print("\n" + "="*60)
    print(f"[AGI COMPLETE] Lifetime Summary:")
    print(f"  Total Episodes: {episodes}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Best Energy Achieved: {agi_system.best_energy:.4f}")
    print(f"  Semantic Knowledge: {len(agi_system.memory.semantic_memory)} rules")
    print(f"  Final Agent Profile: {agi_system.self_model.current_profile}")
    print(f"  Current Truth Vector Norm: {agi_system.values.compute_current_truth().norm():.3f}")
    print("="*60)


if __name__ == '__main__':
    main_agi()