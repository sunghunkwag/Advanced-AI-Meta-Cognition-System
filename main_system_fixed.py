"""Fixed main system with REINFORCE algorithm for proper gradient flow.

Critical fixes:
1. REINFORCE (Policy Gradient) instead of broken gradient flow
2. GPU compatibility (.cpu() added)
3. Correct loss calculation
4. Proper action sampling with log probabilities
"""

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
    print("[INIT] Advanced AI System (FIXED VERSION)")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize World & Perception
    world = World(size=16)
    vision = VisionSystem()
    
    # Initialize Mind with Soul
    LATENT_DIM = 8
    v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
    mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=v_truth)
    mind = mind.to(device)
    
    # Initialize Body
    body = ActionDecoder(latent_dim=LATENT_DIM)
    body = body.to(device)
    
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
    
    # Metrics storage
    episode_rewards = []
    
    # Life Cycle Loop
    for step in range(1, 51):
        try:
            # === PERCEPTION ===
            world_state = world.get_state()
            nodes, adj = vision.perceive(world_state)
            nodes = nodes.to(device)
            adj = adj.to(device)
            
            # === MIND (Reasoning) ===
            z = mind(nodes, adj)
            consistency = mind.check_consistency(z)
            
            # === BODY (Action Selection) ===
            action_logits, params = body(z)
            
            # CURRICULUM: Force DRAW in early steps to bootstrap
            if step <= 10:
                action_logits = torch.zeros_like(action_logits)
                action_logits[0, 0] = 10.0  # Force DRAW
            elif step <= 20:
                if np.random.rand() < 0.5:
                    action_logits = torch.zeros_like(action_logits)
                    action_logits[0, 0] = 10.0
            
            # CRITICAL FIX: Sample action with log probability for REINFORCE
            action, log_prob = body.sample_action(action_logits, params)
            
            # === ACT ON WORLD ===
            world.apply_action(action)
            
            # === CALCULATE ENERGY (Reward) ===
            world_energy = world.calculate_energy()
            
            # === HEART (Emotions) ===
            heart.update(world_energy, consistency.item())
            dopamine, serotonin = heart.get_hormones()
            state_mode = heart.get_state()
            
            # === SOUL (Crystallization Check) ===
            soul.update_state((dopamine, serotonin))
            ewc_loss = soul.ewc_loss(mind)
            
            # === REINFORCE LOSS ===
            # CRITICAL FIX: Policy gradient instead of broken gradient flow
            # Negative reward (we want to minimize energy)
            reward = -world_energy
            
            # Policy gradient loss: -log_prob * reward
            # (Maximize log_prob for good actions)
            policy_loss = -log_prob * reward
            
            # Consistency loss (maximize consistency with truth)
            consistency_loss = 1.0 - consistency
            
            # Total loss
            loss = policy_loss + consistency_loss + ewc_loss
            
            # === LEARNING (Backpropagation) ===
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(mind.parameters()) + list(body.parameters()), 
                max_norm=1.0
            )
            
            optimizer.step()
            
            # Store reward
            episode_rewards.append(reward)
            
            # === LOGGING ===
            if step % 5 == 0 or step == 1:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                print(f"Step {step:02d} | {state_mode:5s} | "
                      f"D:{dopamine:.2f} S:{serotonin:.2f} | "
                      f"E:{world_energy:.4f} R:{reward:.2f} | "
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
    print(f"Average Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
