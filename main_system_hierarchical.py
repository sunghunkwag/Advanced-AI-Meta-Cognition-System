import torch
import torch.optim as optim
import numpy as np
import sys

# Import Core Modules
from vision import VisionSystem
from manifold import GraphAttentionManifold
from energy import NeuroChemicalEngine
from action_decoder import ActionDecoder
from automata import IntrinsicAutomata
from world import World
from soul import get_soul_vectors

# Import NEW Modules - Hierarchical Goal System
from goal_system import GoalSystem, GoalTracker
from environment_manager import EnvironmentManager, AdaptiveStrategy, CurriculumManager

def main():
    print("[INIT] Advanced AI System with Hierarchical Goals")
    print("="*80)
    
    # === NEW: Initialize Goal & Environment Systems ===
    goal_system = GoalSystem()
    goal_tracker = GoalTracker()
    env_manager = EnvironmentManager()
    adaptive_strategy = AdaptiveStrategy()
    curriculum = CurriculumManager()
    
    # === NEW: Curriculum Learning Mode ===
    print(f"[CURRICULUM] Starting at: {curriculum.get_current_stage_name()}")
    
    # Create environment based on curriculum
    env_type = curriculum.get_next_environment()
    world = env_manager.create_environment(env_type)
    
    # Analyze environment and select strategy
    env_analysis = env_manager.analyze_environment(world)
    strategy = adaptive_strategy.select_strategy(env_analysis)
    strategy_config = adaptive_strategy.get_strategy_config(strategy)
    
    print(f"[ENVIRONMENT] Type: {env_type}, Size: {world.size}x{world.size}")
    print(f"[STRATEGY] Selected: {strategy}")
    print(f"[STRATEGY] Description: {strategy_config['description']}")
    print("="*80)
    
    # Initialize Perception
    vision = VisionSystem()
    
    # Initialize Mind with Soul
    LATENT_DIM = 8
    v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
    mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=v_truth)
    
    # Initialize Body
    body = ActionDecoder(latent_dim=LATENT_DIM)
    
    # Initialize Heart & Soul
    heart = NeuroChemicalEngine()
    soul = IntrinsicAutomata(mind)
    
    # Optimizer (with adaptive LR)
    optimizer = optim.Adam(
        list(mind.parameters()) + list(body.parameters()),
        lr=0.01
    )
    
    # Advanced Learning Tracking
    energy_history = []
    action_history = []
    best_energy = float('inf')
    prev_world_state = None
    
    # === NEW: Goal System Tracking ===
    current_meta_goal = None
    current_strategic_goal = None
    steps_on_current_goal = 0
    goal_switch_log = []

    print("[OK] System initialized. Starting life cycle...")
    print("="*80)
    
    # Life Cycle Loop (1000 steps)
    for step in range(1, 1001):
        try:
            # === PERCEPTION ===
            world_state = world.get_state()
            nodes, adj = vision.perceive(world_state)
            
            # === MIND (Reasoning) ===
            z = mind(nodes, adj)
            consistency = mind.check_consistency(z)
            
            # === CALCULATE ENERGY ===
            world_energy = world.calculate_energy()
            
            # === HEART (Emotions) ===
            heart.update(world_energy, consistency.item())
            dopamine, serotonin = heart.get_hormones()
            state_mode = heart.get_state()
            emotions = {'dopamine': dopamine, 'serotonin': serotonin}
            
            # === SOUL (Crystallization Check) ===
            soul.update_state((dopamine, serotonin))
            ewc_loss = soul.ewc_loss(mind)
            
            # ============================================================
            # === NEW: HIERARCHICAL GOAL SYSTEM ===
            # ============================================================
            
            # Check if we should switch goals
            should_switch = (
                step == 1 or  # First step
                step % 50 == 0 or  # Periodic re-evaluation
                (current_strategic_goal and 
                 goal_tracker.should_switch_goal(current_strategic_goal, steps_on_current_goal))
            )
            
            if should_switch:
                # Select new meta-goal
                context = {'step': step, 'energy': world_energy, 'strategy': strategy}
                current_meta_goal = goal_system.select_meta_goal(
                    world_state, 
                    emotions, 
                    context
                )
                
                # Decompose to strategic goals
                strategic_goals = goal_system.decompose_to_strategy(
                    current_meta_goal,
                    world_state,
                    context
                )
                
                # Select top strategic goal
                if strategic_goals:
                    current_strategic_goal = strategic_goals[0]
                    goal_tracker.set_active_goal(current_strategic_goal)
                    steps_on_current_goal = 0
                    
                    goal_switch_log.append({
                        'step': step,
                        'meta_goal': current_meta_goal,
                        'strategic_goal': current_strategic_goal
                    })
            
            steps_on_current_goal += 1
            
            # Evaluate progress on current goal
            if current_strategic_goal:
                progress = goal_tracker.evaluate_progress(
                    current_strategic_goal,
                    world_state,
                    prev_world_state
                )
                
                # Mark completed if progress is high
                if progress >= 0.9:
                    goal_tracker.mark_completed(current_strategic_goal)
            
            # ============================================================
            # === BODY (Goal-Driven Action Selection) ===
            # ============================================================
            action_logits, params = body(z)
            
            # CURRICULUM: Force DRAW in early steps to bootstrap
            if step <= 10:
                action_logits = torch.zeros_like(action_logits)
                action_logits[0, 0] = 10.0  # Force DRAW
            elif step <= 20:
                # 50% forced DRAW
                if np.random.rand() < 0.5:
                    action_logits = torch.zeros_like(action_logits)
                    action_logits[0, 0] = 10.0
            else:
                # === NEW: Goal-Based Action Biasing ===
                if current_strategic_goal:
                    # Get action biases from goal system
                    goal_biases = goal_system.get_action_bias(current_strategic_goal)
                    
                    # Also get strategy biases
                    strategy_biases = strategy_config.get('action_bias', {})
                    
                    # Combine goal and strategy biases (weighted average)
                    combined_biases = {}
                    for action_type in ['DRAW', 'CLEAR', 'NOISE', 'SYMMETRIZE']:
                        goal_weight = goal_biases.get(action_type, 0.25)
                        strategy_weight = strategy_biases.get(action_type, 0.25)
                        combined_biases[action_type] = (goal_weight + strategy_weight) / 2
                    
                    # Apply biases to action logits
                    bias_tensor = torch.tensor([
                        combined_biases.get('DRAW', 0.25),
                        combined_biases.get('CLEAR', 0.25),
                        combined_biases.get('NOISE', 0.25),
                        combined_biases.get('SYMMETRIZE', 0.25)
                    ], dtype=action_logits.dtype)
                    
                    # Add bias (log-space for softmax)
                    action_logits = action_logits + torch.log(bias_tensor + 1e-8)
                
                # === EPSILON-GREEDY EXPLORATION ===
                epsilon = max(0.3 * (1 - step/1000), 0.05)  # Decay 0.3 -> 0.05
                if np.random.rand() < epsilon:
                    # Random action selection
                    action_logits = torch.randn_like(action_logits)
            
            action = body.decode_action(action_logits, params)
            action_history.append(action['type'])
            
            # === ACT ON WORLD ===
            world.apply_action(action)
            prev_world_state = world_state.copy()
            
            # === LEARNING (Backpropagation) ===
            # Loss = World Energy + (1 - Consistency) + EWC
            loss = torch.tensor(world_energy, dtype=torch.float32)
            loss = loss + (1.0 - consistency)
            loss = loss + ewc_loss
            
            # === REWARD SHAPING ===
            # Progress bonus: reward improvement
            energy_history.append(world_energy)
            if len(energy_history) > 10:
                recent_avg = np.mean(energy_history[-10:])
                prev_avg = np.mean(energy_history[-20:-10]) if len(energy_history) >= 20 else recent_avg
                improvement = prev_avg - recent_avg
                if improvement > 0:
                    loss = loss - torch.tensor(improvement * 0.1, dtype=torch.float32)  # Bonus
            
            # === ACTION DIVERSITY BONUS ===
            if len(action_history) >= 20:
                recent_actions = action_history[-20:]
                unique_actions = len(set(recent_actions))
                diversity_bonus = (unique_actions - 2) * 0.02  # Bonus for using 3-4 different actions
                loss = loss - torch.tensor(diversity_bonus, dtype=torch.float32)
            
            # === NEW: GOAL ACHIEVEMENT BONUS ===
            if current_strategic_goal:
                goal_progress = goal_tracker.progress.get(current_strategic_goal, 0.0)
                if goal_progress > 0.5:
                    # Reward making progress on goals
                    goal_bonus = (goal_progress - 0.5) * 0.2
                    loss = loss - torch.tensor(goal_bonus, dtype=torch.float32)
            
            # === ADAPTIVE LEARNING RATE ===
            lr = 0.01 * (0.95 ** (step // 50))  # Exponential decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track best energy
            if world_energy < best_energy:
                best_energy = world_energy
            
            # === LOGGING ===
            if step % 50 == 1 or step <= 50:  # Log every 50 steps or first 50
                goal_info = f"{current_meta_goal[:4].upper() if current_meta_goal else 'NONE'}/{current_strategic_goal[:10] if current_strategic_goal else 'none'}"
                print(f"Step {step:03d} | {state_mode:5s} | "
                      f"D:{dopamine:.2f} S:{serotonin:.2f} | "
                      f"E:{world_energy:.4f} | "
                      f"L:{loss.item():.4f} | "
                      f"LR:{lr:.5f} | "
                      f"Goal:{goal_info:20s} | "
                      f"{action['type']:10s} | "
                      f"Grid:{world.grid.sum():.1f}")
            
            # === CHECK CRYSTALLIZATION ===
            if soul.is_crystallized():
                print("\n" + "="*80)
                print("[NIRVANA] Mind crystallized. Simulation complete.")
                print(f"[GOALS] Completed: {len(goal_tracker.completed_goals)}")
                print(f"[GOALS] Failed: {len(goal_tracker.failed_goals)}")
                print(f"[GOALS] Completion Rate: {goal_tracker.get_completion_rate():.1%}")
                print("="*80)
                break
                
        except Exception as e:
            print(f"\n[ERROR] Step {step} failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*80)
    print(f"[DONE] Completed {step} steps")
    print(f"Final Energy: {world_energy:.4f}")
    print(f"Final Grid Sum: {world.grid.sum():.2f}")
    print(f"Best Energy Achieved: {best_energy:.4f}")
    print(f"\n[GOAL SYSTEM STATS]")
    print(f"  Meta-Goal Switches: {len(goal_switch_log)}")
    print(f"  Completed Goals: {len(goal_tracker.completed_goals)}")
    print(f"  Active Goals: {len(goal_tracker.active_goals)}")
    print(f"  Goal Completion Rate: {goal_tracker.get_completion_rate():.1%}")
    print("="*80)

if __name__ == "__main__":
    main()
