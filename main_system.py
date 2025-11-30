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
    
    # Optimizer
    optimizer = optim.Adam(
        list(mind.parameters()) + list(body.parameters()),
        lr=0.01
    )

    print("[OK] System initialized. Starting life cycle...")
    print("="*60)
    
    # Life Cycle Loop
    for step in range(1, 51):
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
            
            action = body.decode_action(action_logits, params)
            
            # === ACT ON WORLD ===
            world.apply_action(action)
            
            # === LEARNING (Backpropagation) ===
            # Loss = World Energy + (1 - Consistency) + EWC
            loss = torch.tensor(world_energy, dtype=torch.float32)
            loss = loss + (1.0 - consistency)
            loss = loss + ewc_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # === LOGGING ===
            print(f"Step {step:02d} | {state_mode:5s} | "
                  f"D:{dopamine:.2f} S:{serotonin:.2f} | "
                  f"E:{world_energy:.4f} | "
                  f"L:{loss.item():.4f} | "
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
