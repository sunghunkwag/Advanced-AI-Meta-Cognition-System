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
    mind = GraphAttentionManifold(
        nfeat=3,
        nhid=16,
        nclass=LATENT_DIM,
        truth_vector=v_truth,
        reject_vector=v_rej
    )
    
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
    consistency_history = []
    hormone_history = []
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

            energy_history.append(world_energy)
            consistency_history.append(consistency.item())
            
            # === HEART (Emotions) ===
            density = np.sum(world_state > 0.1) / world_state.size
            h_sym = 1.0 - np.abs(world_state - np.fliplr(world_state)).mean()
            v_sym = 1.0 - np.abs(world_state - np.flipud(world_state)).mean()
            symmetry = (h_sym + v_sym) / 2.0
            prediction_error = 1.0 - consistency.item()

            heart.update(world_energy, consistency.item(), density, symmetry, prediction_error)
            hormones = heart.get_hormones()
            dopamine = hormones['dopamine']
            serotonin = hormones['serotonin']
            cortisol = hormones['cortisol']

            # === SOUL (Crystallization Check) ===
            hormone_history.append((dopamine, serotonin))
            soul.update_state(
                (dopamine, serotonin),
                nodes,
                adj,
                energy_history=energy_history,
                consistency_history=consistency_history,
                hormone_history=hormone_history,
            )
            ewc_loss = soul.ewc_loss(mind)
            
            # === BODY (Action Selection) ===
            action_logits, params = body(z)
            
            # CURRICULUM: Extended bootstrap phase
            if step <= 30:
                # Phase 1: Pure DRAW (0-20)
                if step <= 20:
                    action_logits = torch.zeros_like(action_logits)
                    action_logits[0, 0] = 10.0  # Force DRAW
                # Phase 2: DRAW + SYMMETRIZE (21-30)
                else:
                    if np.random.rand() < 0.7:
                        action_logits = torch.zeros_like(action_logits)
                        if np.random.rand() < 0.5:
                            action_logits[0, 0] = 10.0
                        else:
                            action_logits[0, 1] = 10.0
            elif step <= 50:
                # Phase 3: Guided exploration (31-50)
                if np.random.rand() < 0.5:
                    action_logits = torch.zeros_like(action_logits)
                    action_logits[0, np.random.choice([0, 1, 2])] = 10.0
            else:
                # === EPSILON-GREEDY EXPLORATION ===
                epsilon = max(0.3 * (1 - (step-50)/950), 0.05)
                if np.random.rand() < epsilon:
                    # Random action selection
                    action_logits = torch.randn_like(action_logits)
            
            action = body.decode_action(action_logits, params)
            action_history.append(action['type'])
            
            # === ACT ON WORLD ===
            world.apply_action(action)
            
            # === LEARNING (Backpropagation) ===
            consistency = mind.check_consistency(z)
            rejection_penalty = mind.check_rejection(z)

            loss = torch.tensor(world_energy, dtype=torch.float32)
            loss = loss + (1.0 - consistency)
            loss = loss + (rejection_penalty * 0.5)
            loss = loss + ewc_loss
            
            # === REWARD SHAPING ===
            # Progress bonus: reward improvement
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
            
            # === ADAPTIVE LEARNING RATE WITH WARMUP ===
            warmup_steps = 20
            if step <= warmup_steps:
                lr = 0.001 + (0.01 - 0.001) * (step / warmup_steps)
            else:
                lr = 0.01 * (0.95 ** ((step - warmup_steps) // 50))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Safety check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[ERROR] Invalid loss at step {step}: {loss.item()}")
                print(f"  - world_energy: {world_energy}")
                print(f"  - consistency: {consistency.item()}")
                print(f"  - ewc_loss: {ewc_loss.item()}")
                # Skip this step
                continue

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(mind.parameters()) + list(body.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            # Track best energy
            if world_energy < best_energy:
                best_energy = world_energy
            
            # === LOGGING ===
            if step % 50 == 1 or step <= 50:
                print(f"Step {step:04d} | "
                      f"E:{world_energy:.4f} | "
                      f"Cons:{consistency.item():.3f} | "
                      f"D:{dopamine:.2f} S:{serotonin:.2f} C:{cortisol:.2f} | "
                      f"L:{loss.item():.4f} | "
                      f"LR:{lr:.5f} | "
                      f"Act:{action['type']:10s} | "
                      f"Grid:Σ={world.grid.sum():.1f} Max={world.grid.max():.2f}")

                if step % 100 == 0:
                    print(f"  └─ Details: Density={density:.3f}, Symmetry={symmetry:.3f}, "
                          f"PredErr={prediction_error:.3f}")
                    if soul.is_crystallized():
                        print(f"  └─ [SOUL] Crystallized (EWC Active)")

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