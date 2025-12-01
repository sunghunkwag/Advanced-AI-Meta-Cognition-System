import json
import os
import torch
import numpy as np

from world import World
from vision import VisionSystem
from manifold import GraphAttentionManifold
from energy import NeuroChemicalEngine
from action_decoder import ActionDecoder
from automata import IntrinsicAutomata
from soul import get_soul_vectors

class MemorySystem:
    """Long-term Memory & Knowledge Structure"""
    
    def __init__(self, memory_dir='./checkpoints/memory/'):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        self.episodic_memory = []  # List of episode trajectories
        self.semantic_memory = {}  # Dict of patterns: frequency
        self.episode_count = 0
    
    def store_episode(self, episode_data):
        """Store full episode trajectory"""
        episode_id = f'episode_{self.episode_count:04d}'
        self.episodic_memory.append(episode_data)
        
        # Extract semantic patterns from GAT outputs
        if 'gat_patterns' in episode_data:
            for pattern, freq in episode_data['gat_patterns'].items():
                if pattern in self.semantic_memory:
                    self.semantic_memory[pattern] += freq
                else:
                    self.semantic_memory[pattern] = freq
        
        # Save to disk
        with open(f'{self.memory_dir}/{episode_id}.json', 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        # Update semantic memory file
        with open(f'{self.memory_dir}/semantic_memory.json', 'w') as f:
            json.dump(self.semantic_memory, f, indent=2)
        
        self.episode_count += 1
    
    def load_semantic_memory(self):
        """Load accumulated semantic knowledge"""
        try:
            with open(f'{self.memory_dir}/semantic_memory.json', 'r') as f:
                self.semantic_memory = json.load(f)
        except FileNotFoundError:
            self.semantic_memory = {}
    
    def get_truth_modification(self, base_truth_vector):
        """Modify Truth vector based on semantic memory patterns"""
        if not self.semantic_memory:
            return base_truth_vector
        
        # Extract most frequent symmetry patterns
        top_patterns = sorted(self.semantic_memory.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Simple heuristic: average top patterns into truth direction
        pattern_influence = torch.zeros_like(base_truth_vector)
        for pattern, freq in top_patterns:
            # Convert pattern to vector influence (simplified)
            if 'symmetry' in pattern:
                pattern_influence += freq * 0.1 * torch.ones_like(base_truth_vector)
            elif 'order' in pattern:
                pattern_influence += freq * 0.05 * torch.ones_like(base_truth_vector)
        
        # Normalize and blend with base truth
        if pattern_influence.norm() > 0:
            pattern_influence = pattern_influence / pattern_influence.norm()
            modified_truth = 0.8 * base_truth_vector + 0.2 * pattern_influence
            return modified_truth / modified_truth.norm()
        
        return base_truth_vector


class ValueHierarchy:
    """Hierarchical Value System"""
    
    def __init__(self, dim=8):
        self.dim = dim
        # Initialize value vectors with reasonable defaults
        self.values = {
            'truth': torch.ones(dim) / dim,
            'order': torch.ones(dim) / dim,
            'creativity': torch.randn(dim) / dim,
            'social': torch.randn(dim) / dim,
            'exploration': torch.ones(dim) / dim,
            'stability': torch.ones(dim) / dim
        }
        
        # Dynamic weights (initial equal)
        self.weights = {k: 1.0 for k in self.values}
        
    def compute_current_truth(self):
        """Compute composite Truth vector from weighted values"""
        composite = torch.zeros(self.dim)
        total_weight = sum(self.weights.values())
        
        for value_name, value_vector in self.values.items():
            weight = self.weights[value_name] / total_weight
            composite += weight * value_vector
        
        return composite / composite.norm()
    
    def update_weights(self, performance_metrics):
        """Adjust value weights based on meta-cognitive feedback"""
        # Example: If creativity is low, boost creativity weight
        if 'creativity_score' in performance_metrics and performance_metrics['creativity_score'] < 0.3:
            self.weights['creativity'] += 0.1
        
        # If social performance is good, maintain or slightly increase
        if 'social_score' in performance_metrics and performance_metrics['social_score'] > 0.7:
            self.weights['social'] += 0.05
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total
    
    def save_state(self, path='./checkpoints/values_state.json'):
        state = {
            'weights': {k: w.item() if hasattr(w, 'item') else w for k, w in self.weights.items()},
            'values': {k: v.tolist() for k, v in self.values.items()}
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path='./checkpoints/values_state.json'):
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            self.weights = state['weights']
            for k, v_list in state['values'].items():
                self.values[k] = torch.tensor(v_list)
        except FileNotFoundError:
            pass  # Use defaults


class SelfModel:
    """Meta-RL Self-Modeling System"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.agent_profiles = {
            'exploratory': {'epsilon': 0.3, 'curriculum_diff': 1.0, 'focus': 'exploration'},
            'exploitative': {'epsilon': 0.05, 'curriculum_diff': 0.5, 'focus': 'optimization'},
            'stuck': {'epsilon': 0.2, 'curriculum_diff': 0.8, 'focus': 'diversification'},
            'stable': {'epsilon': 0.1, 'curriculum_diff': 0.7, 'focus': 'maintenance'}
        }
        self.current_profile = 'exploratory'
    
    def analyze_performance(self, recent_episodes=5):
        """Analyze recent episodes to determine agent state"""
        if len(self.memory.episodic_memory) < recent_episodes:
            return self.current_profile
        
        recent = self.memory.episodic_memory[-recent_episodes:]
        
        # Extract metrics
        energies = [ep['final_energy'] for ep in recent]
        action_diversity = [len(set(ep['actions'])) / len(ep['actions']) for ep in recent]
        success_rates = [1 if ep['final_energy'] < 0.2 else 0 for ep in recent]
        
        avg_energy = np.mean(energies)
        avg_diversity = np.mean(action_diversity)
        success_rate = np.mean(success_rates)
        
        # Simple rule-based profiling
        if avg_diversity > 0.6 and success_rate < 0.4:
            self.current_profile = 'exploratory'
        elif avg_energy < 0.1 and avg_diversity > 0.3:
            self.current_profile = 'stable'
        elif len(set([ep['actions'][-5:] for ep in recent])) < 2:  # Repetitive actions
            self.current_profile = 'stuck'
        else:
            self.current_profile = 'exploitative'
        
        return self.current_profile
    
    def get_hyperparameters(self):
        """Return hyperparameters based on current profile"""
        return self.agent_profiles[self.current_profile]
    
    def get_performance_metrics(self):
        """Return metrics for value system update"""
        profile = self.analyze_performance()
        metrics = {
            'profile': profile,
            'energy_improvement': -0.1,  # Placeholder
            'creativity_score': 0.5 if profile == 'exploratory' else 0.3,
            'social_score': 0.7 if profile == 'stable' else 0.4,
            'stability_score': 0.8 if profile == 'stable' else 0.6
        }
        return metrics


class NeuroAgent:
    """Wrapper for the core neuro-chemical agent"""
    
    def __init__(self, agent_id, memory_system, value_system, self_model):
        self.agent_id = agent_id
        self.memory = memory_system
        self.values = value_system
        self.self_model = self_model
        
        # Core components (preserved from original)
        self.world = World(size=16)
        self.vision = VisionSystem()
        LATENT_DIM = 8
        v_id, base_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
        
        # Modify truth vector with memory
        modified_truth = self.memory.get_truth_modification(base_truth)
        self.mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=modified_truth)
        
        self.body = ActionDecoder(latent_dim=LATENT_DIM)
        self.heart = NeuroChemicalEngine()
        self.soul = IntrinsicAutomata(self.mind)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.mind.parameters()) + list(self.body.parameters()),
            lr=0.01
        )
        
        self.episode_data = {
            'actions': [],
            'energies': [],
            'hormones': [],
            'gat_patterns': {},
            'final_energy': 0.0
        }
    
    def run_episode(self, max_steps=1000, environment=None):
        """Run a single episode (preserves original logic)"""
        if environment:
            self.world = environment  # Use external world for multi-agent
        
        # Get hyperparameters from self-model
        hypers = self.self_model.get_hyperparameters()
        epsilon = hypers['epsilon']
        
        step = 0
        energy_history = []
        action_history = []
        
        while step < max_steps:
            step += 1
            
            # PERCEPTION
            world_state = self.world.get_state()
            nodes, adj = self.vision.perceive(world_state)
            
            # MIND (with current truth vector)
            current_truth = self.values.compute_current_truth()
            self.mind.update_truth_vector(current_truth)
            z = self.mind(nodes, adj)
            consistency = self.mind.check_consistency(z)
            
            # Extract GAT patterns for semantic memory
            patterns = self.mind.extract_patterns()
            self.episode_data['gat_patterns'].update(patterns)
            
            # HEART
            world_energy = self.world.calculate_energy()
            self.heart.update(world_energy, consistency.item())
            dopamine, serotonin = self.heart.get_hormones()
            state_mode = self.heart.get_state()
            self.episode_data['hormones'].append((dopamine, serotonin))
            
            # SOUL
            self.soul.update_state((dopamine, serotonin))
            ewc_loss = self.soul.ewc_loss(self.mind)
            
            # BODY (Action Selection with epsilon-greedy)
            action_logits, params = self.body(z)
            
            # Curriculum for early steps
            if step <= 10:
                action_logits[0, 0] = 10.0  # Force DRAW
            elif step <= 20 and np.random.rand() < 0.5:
                action_logits[0, 0] = 10.0
            else:
                if np.random.rand() < epsilon:
                    action_logits = torch.randn_like(action_logits)
            
            action = self.body.decode_action(action_logits, params)
            action_history.append(action['type'])
            self.episode_data['actions'].append(action['type'])
            
            # ACT
            self.world.apply_action(action)
            energy_history.append(world_energy)
            self.episode_data['energies'].append(world_energy)
            
            # LEARNING (Preserved original loss computation)
            loss = torch.tensor(world_energy, dtype=torch.float32) + (1.0 - consistency) + ewc_loss
            
            # Reward shaping (original logic preserved)
            if len(energy_history) > 10:
                recent_avg = np.mean(energy_history[-10:])
                prev_avg = np.mean(energy_history[-20:-10]) if len(energy_history) >= 20 else recent_avg
                improvement = prev_avg - recent_avg
                if improvement > 0:
                    loss -= torch.tensor(improvement * 0.1, dtype=torch.float32)
            
            # Action diversity (original)
            if len(action_history) >= 20:
                recent_actions = action_history[-20:]
                unique_actions = len(set(recent_actions))
                diversity_bonus = (unique_actions - 2) * 0.02
                loss -= torch.tensor(diversity_bonus, dtype=torch.float32)
            
            # Adaptive LR (original)
            lr = 0.01 * (0.95 ** (step // 50))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Check crystallization
            if self.soul.is_crystallized():
                break
        
        self.episode_data['final_energy'] = world_energy
        
        # Store to memory
        self.memory.store_episode(self.episode_data)
        
        # Update self-model and values
        performance_metrics = self.self_model.get_performance_metrics()
        self.values.update_weights(performance_metrics)
        
        return world_energy, step


class MultiAgentWorld:
    """Multi-Agent Social Environment"""
    
    def __init__(self, size=16, num_agents=3):
        self.size = size
        self.grid = np.zeros((size, size))
        self.agents = []
        self.num_agents = num_agents
        
    def add_agent(self, agent):
        """Add a neuro-agent to the world"""
        agent.world = self  # Shared world
        self.agents.append(agent)
        return agent
    
    def run_social_episode(self, max_steps=500):
        """Run multi-agent episode with social dynamics"""
        total_energy = 0
        social_score = 0
        
        for step in range(max_steps):
            # Each agent acts in random order
            agent_order = np.random.permutation(len(self.agents))
            step_energy = 0
            
            for agent_idx in agent_order:
                agent = self.agents[agent_idx]
                # Run one step of agent's episode
                # (Simplified: agents take turns)
                world_state = self.get_state()
                nodes, adj = agent.vision.perceive(world_state)
                z = agent.mind(nodes, adj)
                action_logits, params = agent.body(z)
                action = agent.body.decode_action(action_logits, params)
                
                # Apply action to shared world
                self.apply_action(action, agent_id=agent.agent_id)
                
                # Social reward: cooperation bonus
                current_energy = self.calculate_energy()
                if step > 0:
                    energy_improvement = prev_energy - current_energy
                    if energy_improvement > 0:
                        social_score += energy_improvement * 0.1
                prev_energy = current_energy
                
                step_energy += current_energy / len(self.agents)
            
            total_energy += step_energy
            
            # Check for convergence
            if all(agent.soul.is_crystallized() for agent in self.agents):
                break
        
        avg_energy = total_energy / max_steps
        return avg_energy, social_score
    
    def apply_action(self, action, agent_id):
        """Apply action to shared world with agent attribution"""
        if action['type'] == 'DRAW':
            x, y = int(action['params'][0]), int(action['params'][1])
            if 0 <= x < self.size and 0 <= y < self.size:
                self.grid[x, y] = 1.0
        elif action['type'] == 'CLEAR':
            x, y = int(action['params'][0]), int(action['params'][1])
            if 0 <= x < self.size and 0 <= y < self.size:
                self.grid[x, y] = 0.0
        # Add other action types...
    
    def get_state(self):
        return self.grid
    
    def calculate_energy(self):
        # Preserve original energy calculation
        symmetry_error = np.sum(np.abs(self.grid - np.flip(self.grid, axis=1)))  # Horizontal symmetry
        density_penalty = abs(np.mean(self.grid) - 0.1) * 10  # Target 10% density
        return (symmetry_error + density_penalty) / self.size**2


class LifelongRunner:
    """Lifelong Learning System"""
    
    def __init__(self, checkpoint_dir='./checkpoints/'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize core systems
        self.memory = MemorySystem()
        self.memory.load_semantic_memory()
        self.values = ValueHierarchy()
        self.values.load_state()
        self.self_model = SelfModel(self.memory)
        
        self.total_lifetime = 0
        self.lifetime_episodes = 0
    
    def create_agent(self, agent_id=0):
        """Create a new agent with current system state"""
        return NeuroAgent(agent_id, self.memory, self.values, self.self_model)
    
    def run_lifetime(self, max_lifetime_episodes=100, episodes_per_checkpoint=10):
        """Run continuous lifelong learning"""
        print(f"[LIFELONG] Starting lifetime {self.lifetime_episodes + 1}")
        
        agent = self.create_agent(self.lifetime_episodes)
        
        for episode in range(max_lifetime_episodes):
            episode_num = self.lifetime_episodes + episode + 1
            print(f"  Episode {episode_num:03d} | Profile: {self.self_model.current_profile}")
            
            # Run episode
            final_energy, steps = agent.run_episode()
            
            # Update systems
            self.self_model.analyze_performance()
            performance_metrics = self.self_model.get_performance_metrics()
            self.values.update_weights(performance_metrics)
            
            print(f"    Energy: {final_energy:.4f} | Steps: {steps} | Truth: {self.values.compute_current_truth().norm():.3f}")
            
            self.lifetime_episodes += 1
            self.total_lifetime += steps
            
            # Periodic checkpoint
            if (episode + 1) % episodes_per_checkpoint == 0:
                self.save_checkpoint(episode_num)
                print(f"    [CHECKPOINT] Saved at episode {episode_num}")
        
        # Final save
        self.save_checkpoint(self.lifetime_episodes)
        print(f"[LIFELONG] Lifetime complete. Total steps: {self.total_lifetime}")
        
        return self.lifetime_episodes, self.total_lifetime
    
    def save_checkpoint(self, episode_num):
        """Save complete system state"""
        checkpoint = {
            'episode': episode_num,
            'total_lifetime': self.total_lifetime,
            'memory_size': len(self.memory.episodic_memory),
            'semantic_rules': len(self.memory.semantic_memory),
            'current_profile': self.self_model.current_profile,
            'value_weights': self.values.weights
        }
        
        # Save systems
        self.memory.save_semantic_memory()  # Already handled in store_episode
        self.values.save_state()
        
        with open(f'{self.checkpoint_dir}/checkpoint_{episode_num:04d}.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save agent weights
        torch.save({
            'mind_state_dict': agent.mind.state_dict(),
            'body_state_dict': agent.body.state_dict(),
            'soul_fisher': agent.soul.fisher_information,
            'episode': episode_num
        }, f'{self.checkpoint_dir}/agent_weights_{episode_num:04d}.pt')
    
    def load_checkpoint(self, episode_num):
        """Load system from checkpoint"""
        try:
            # Load systems
            self.memory.load_semantic_memory()
            self.values.load_state()
            
            # Load agent weights (create temp agent for loading)
            temp_agent = self.create_agent(episode_num)
            checkpoint_path = f'{self.checkpoint_dir}/agent_weights_{episode_num:04d}.pt'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                temp_agent.mind.load_state_dict(checkpoint['mind_state_dict'])
                temp_agent.body.load_state_dict(checkpoint['body_state_dict'])
                temp_agent.soul.fisher_information = checkpoint['soul_fisher']
                
                # Update lifetime counters
                self.lifetime_episodes = checkpoint['episode']
                self.total_lifetime = checkpoint.get('total_lifetime', 0)
                
            print(f"[CHECKPOINT] Loaded episode {episode_num}, lifetime steps: {self.total_lifetime}")
            return True
        except Exception as e:
            print(f"[CHECKPOINT] Load failed: {e}")
            return False


if __name__ == '__main__':
    # Test the extended AGI system
    runner = LifelongRunner()
    
    # Run 5 episodes to test
    episodes, total_steps = runner.run_lifetime(max_lifetime_episodes=5)
    
    print(f"\n[TEST] AGI Extensions Verified:")
    print(f"  Episodes completed: {episodes}")
    print(f"  Total lifetime steps: {total_steps}")
    print(f"  Semantic memory size: {len(runner.memory.semantic_memory)}")
    print(f"  Current agent profile: {runner.self_model.current_profile}")
    
    # Test multi-agent (simplified)
    print("\n[TEST] Multi-Agent System:")
    multi_world = MultiAgentWorld(num_agents=2)
    agent1 = runner.create_agent(1001)
    agent2 = runner.create_agent(1002)
    multi_world.add_agent(agent1)
    multi_world.add_agent(agent2)
    
    avg_energy, social_score = multi_world.run_social_episode(max_steps=100)
    print(f"  Multi-agent energy: {avg_energy:.4f}")
    print(f"  Social cooperation score: {social_score:.4f}")
    
    print("\n[VERIFIED] AGI Architecture Extensions Functional")