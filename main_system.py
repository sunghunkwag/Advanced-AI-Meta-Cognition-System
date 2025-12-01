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

def main():
    print("[INIT] Advanced AI System")
    print("="*60)
    
    # Initialize World & Perception
    world = World(size=16)
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

    print("[OK] System initialized. Starting life cycle...")
    print("="*60)
    
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
            
            # === SOUL (Crystallization Check) ===
            soul.update_state((dopamine, serotonin))
            ewc_loss = soul.ewc_loss(mind)
            
            # === BODY (Action Selection) ===
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
                # === EPSILON-GREEDY EXPLORATION ===
                epsilon = max(0.3 * (1 - step/1000), 0.05)  # Decay 0.3 -> 0.05
                if np.random.rand() < epsilon:
                    # Random action selection
                    action_logits = torch.randn_like(action_logits)
            
            action = body.decode_action(action_logits, params)
            action_history.append(action['type'])
            
            # === ACT ON WORLD ===
            world.apply_action(action)
            
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
                print(f"Step {step:03d} | {state_mode:5s} | "
                      f"D:{dopamine:.2f} S:{serotonin:.2f} | "
                      f"E:{world_energy:.4f} | "
                      f"L:{loss.item():.4f} | "
                      f"LR:{lr:.5f} | "
                      f"{action['type']:10s} | "
                      f"Grid:{world.grid.sum():.1f}")
            
            # === CHECK CRYSTALLIZATION ===
            if soul.is_crystallized():
                print("\n" + "="*60)
                print("[NIRVANA] Mind crystallized. Simulation complete.")
                print("="*60)
                break
                
        except Exception as e:
            print(f"\n[ERROR] Step {step} failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*60)
    print(f"[DONE] Completed {step} steps")
    print(f"Final Energy: {world_energy:.4f}")
    print(f"Final Grid Sum: {world.grid.sum():.2f}")
    print("="*60)

if __name__ == "__main__":
    main()