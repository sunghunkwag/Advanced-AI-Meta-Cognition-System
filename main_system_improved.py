import torch
import torch.optim as optim
import numpy as np

# Core Modules
from vision import VisionSystem
from manifold import GraphAttentionManifold
from energy import NeuroChemicalEngine
from action_decoder import ActionDecoder
from automata import IntrinsicAutomata
from world import World
from soul import get_soul_vectors

# Hierarchical Goal System
from goal_system import GoalSystem, GoalTracker
from environment_manager import EnvironmentManager, AdaptiveStrategy
from curiosity import CuriosityModule, HierarchicalLossCalculator

def main():
    print("[INIT] Advanced AI with IMPROVED Goal System v2.0")
    print("="*80)
    
    # Initialize Core Systems
    env_manager = EnvironmentManager()
    goal_system = GoalSystem()
    goal_tracker = GoalTracker()
    adaptive_strategy = AdaptiveStrategy()
    
    # NEW: Curiosity & Hierarchical Loss
    curiosity = CuriosityModule()
    hierarchical_loss_calc = HierarchicalLossCalculator()
    
    # Start with 8x8 grid for easier learning
    world = env_manager.create_environment("tiny")  # 8x8
    env_analysis = env_manager.analyze_environment(world)
    strategy = adaptive_strategy.select_strategy(env_analysis)
    
    print(f"[ENV] Size: {world.size}x{world.size} (tiny)")
    print(f"[STRATEGY] {strategy}")
    print("="*80)
    
    # Initialize Components
    vision = VisionSystem()
    LATENT_DIM = 8
    v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
    mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=v_truth)
    body = ActionDecoder(latent_dim=LATENT_DIM)
    heart = NeuroChemicalEngine()
    soul = IntrinsicAutomata(mind)
    
    optimizer = optim.Adam(
        list(mind.parameters()) + list(body.parameters()),
        lr=0.01
    )
    
    # Tracking
    energy_history = []
    action_history = []
    best_energy = float('inf')
    prev_world_state = None
    predicted_next_state = None
    
    # Goal tracking
    current_meta_goal = None
    current_strategic_goal = None
    steps_on_current_goal = 0
    
    print("[OK] All systems initialized. Starting...")
    print("="*80)
    
    # Main Loop
    for step in range(1, 501):  # Reduced to 500 for faster testing
        try:
            # === PERCEPTION ===
            world_state = world.get_state()
            nodes, adj = vision.perceive(world_state)
            
            # === MIND ===
            z = mind(nodes, adj)
            consistency = mind.check_consistency(z)
            
            # === ENERGY ===
            world_energy = world.calculate_energy()
            
            # === HEART ===
            heart.update(world_energy, consistency.item())
            dopamine, serotonin = heart.get_hormones()
            emotions = {'dopamine': dopamine, 'serotonin': serotonin}
            
            # === SOUL ===
            soul.update_state((dopamine, serotonin))
            ewc_loss = soul.ewc_loss(mind)
            
            # ============================================================
            # === GOAL SELECTION (with better persistence) ===
            # ============================================================
            should_switch = (
                step == 1 or
                (current_strategic_goal and 
                 goal_tracker.should_switch_goal(current_strategic_goal, steps_on_current_goal))
            )
            
            if should_switch:
                context = {'step': step, 'energy': world_energy}
                current_meta_goal = goal_system.select_meta_goal(world_state, emotions, context)
                strategic_goals = goal_system.decompose_to_strategy(current_meta_goal, world_state, context)
                
                if strategic_goals:
                    current_strategic_goal = strategic_goals[0]
                    goal_tracker.set_active_goal(current_strategic_goal)
                    steps_on_current_goal = 0
                    print(f"[GOAL] New: {current_meta_goal}/{current_strategic_goal}")
            
            steps_on_current_goal += 1
            
            # Evaluate progress
            strategic_progress = 0.0
            if current_strategic_goal:
                strategic_progress = goal_tracker.evaluate_progress(
                    current_strategic_goal, world_state, prev_world_state
                )
            
            # ============================================================
            # === ACTION SELECTION (with STRONG goal biasing) ===
            # ============================================================
            action_logits, params = body(z)
            
            if step <= 10:
                action_logits = torch.zeros_like(action_logits)
                action_logits[0, 0] = 10.0  # Force DRAW
            else:
                # === STRONG GOAL-BASED BIASING ===
                if current_strategic_goal:
                    goal_biases = goal_system.get_action_bias(current_strategic_goal)
                    
                    # Apply STRONG biasing (power transformation)
                    bias_strength = 3.0  # Strong influence
                    bias_tensor = torch.tensor([
                        goal_biases['DRAW'] ** bias_strength,
                        goal_biases['CLEAR'] ** bias_strength,
                        goal_biases['NOISE'] ** bias_strength,
                        goal_biases['SYMMETRIZE'] ** bias_strength
                    ], dtype=action_logits.dtype)
                    
                    # Mix: 30% original, 70% goal-driven
                    action_logits = action_logits * 0.3 + torch.log(bias_tensor + 1e-8) * 0.7
                
                # Reduced epsilon for more goal-following
                epsilon = max(0.2 * (1 - step/500), 0.03)
                if np.random.rand() < epsilon:
                    action_logits = torch.randn_like(action_logits)
            
            action = body.decode_action(action_logits, params)
            action_history.append(action['type'])
            
            # === CURIOSITY: Predict before acting ===
            if step > 10:
                predicted_next_state = curiosity.predict_next_state(world_state, action['type'])
            
            # === ACT ===
            world.apply_action(action)
            new_world_state = world.get_state()
            
            # === CURIOSITY BONUS ===
            curiosity_bonus = 0.0
            if predicted_next_state is not None:
                curiosity_bonus = curiosity.calculate_curiosity_bonus(predicted_next_state, new_world_state)
            
            # ============================================================
            # === HIERARCHICAL LOSS CALCULATION ===
            # ============================================================
            # Calculate all losses as floats first for logging/tracking
            base_loss_float, loss_breakdown = hierarchical_loss_calc.calculate_total_loss(
                world_energy,
                consistency.item(),
                current_meta_goal,
                current_strategic_goal,
                strategic_progress,
                new_world_state
            )
            
            # Reconstruct differentiable loss
            # 1. Truth loss (from consistency tensor)
            truth_weight = 0.20 # Must match HierarchicalLossCalculator
            truth_loss_tensor = (1.0 - consistency) * truth_weight
            
            # 2. Non-differentiable parts (treated as constants)
            non_diff_loss = base_loss_float - loss_breakdown['truth']
            
            # Final loss with gradients
            loss = truth_loss_tensor + ewc_loss + torch.tensor(non_diff_loss, dtype=torch.float32)
            
            # Curiosity bonus (scalar, affects loss magnitude but not gradients)
            loss = loss - torch.tensor(curiosity_bonus * 0.3, dtype=torch.float32)
            
            # === ADAPTIVE LR ===
            lr = 0.01 * (0.95 ** (step // 50))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track
            energy_history.append(world_energy)
            if world_energy < best_energy:
                best_energy = world_energy
            prev_world_state = world_state.copy()
            
            # === LOGGING ===
            if step % 25 == 1 or step <= 50:
                goal_str = f"{current_meta_goal[:4] if current_meta_goal else 'NONE'}/{current_strategic_goal[:8] if current_strategic_goal else 'none'}"
                print(f"S{step:03d} | E:{world_energy:.3f} | L:{loss.item():.3f} | "
                      f"Goal:{goal_str:15s} | Prog:{strategic_progress:.2f} | "
                      f"Curio:{curiosity_bonus:.2f} | {action['type']:8s}")
            
            # === CRYSTALLIZATION ===
            if soul.is_crystallized():
                print("\n" + "="*80)
                print(f"[NIRVANA] Crystallized at step {step}!")
                print(f"[GOALS] Completed: {len(goal_tracker.completed_goals)}")
                print("="*80)
                break
                
        except Exception as e:
            print(f"\n[ERROR] Step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*80)
    print(f"[DONE] Completed {step} steps")
    print(f"Final Energy: {world_energy:.4f}")
    print(f"Best Energy: {best_energy:.4f}")
    print(f"Grid Sum: {world.grid.sum():.2f}")
    print(f"Goals Completed: {len(goal_tracker.completed_goals)}")
    print(f"Goal Completion Rate: {goal_tracker.get_completion_rate():.1%}")
    print("="*80)

if __name__ == "__main__":
    main()
