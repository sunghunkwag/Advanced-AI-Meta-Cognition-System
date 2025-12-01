import torch
import torch.optim as optim
import numpy as np
import time

# Core Modules
from vision import VisionSystem
from manifold import GraphAttentionManifold
from energy import NeuroChemicalEngine
from action_decoder import ActionDecoder
from automata import IntrinsicAutomata
from world import World
from soul import get_soul_vectors

# New Modules
from environment_manager import EnvironmentManager
from cortex import PrefrontalCortex

def main():
    print("[INIT] Advanced AI: Dual-Process Architecture (Instinct vs Reason)")
    print("="*80)
    
    # Initialize Systems
    env_manager = EnvironmentManager()
    world = env_manager.create_environment("tiny") # 8x8
    
    # System 1: The Heart (Hormones)
    heart = NeuroChemicalEngine()
    
    # System 2: The Cortex (Reason)
    cortex = PrefrontalCortex()
    
    # Body & Mind
    vision = VisionSystem()
    LATENT_DIM = 8
    v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
    mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=v_truth)
    body = ActionDecoder(latent_dim=LATENT_DIM)
    soul = IntrinsicAutomata(mind)
    
    optimizer = optim.Adam(list(mind.parameters()) + list(body.parameters()), lr=0.01)
    
    # Tracking
    history = {
        'energy': [], 'dopamine': [], 'serotonin': [], 'cortisol': [],
        'willpower': [], 'actions': []
    }
    
    print("[OK] Systems Online. Starting Life Cycle...")
    print("="*80)
    
    prev_hormones = {'dopamine': 0.5, 'serotonin': 0.5, 'cortisol': 0.0}
    
    for step in range(1, 1001):
        try:
            # 1. PERCEPTION
            world_state = world.get_state()
            nodes, adj = vision.perceive(world_state)
            
            # 2. MIND (System 1 "Habit")
            z = mind(nodes, adj)
            consistency = mind.check_consistency(z)
            
            # Network proposes an action (The "Habit")
            action_logits, params = body(z)
            network_action = body.decode_action(action_logits, params)
            
            # 3. HEART (System 1 "Instinct")
            # Calculate environmental stats for hormones
            world_energy = world.calculate_energy()
            density = np.sum(world_state > 0.1) / world_state.size
            
            # Symmetry calculation
            h_sym = 1.0 - np.abs(world_state - np.fliplr(world_state)).mean()
            v_sym = 1.0 - np.abs(world_state - np.flipud(world_state)).mean()
            symmetry = (h_sym + v_sym) / 2
            
            # Prediction Error (Simulated for now, or use z difference)
            # For now, use consistency as a proxy for "understanding" (High consistency = Low error)
            prediction_error = 1.0 - consistency.item()
            
            # Update Hormones
            heart.update(world_energy, consistency.item(), density, symmetry, prediction_error)
            hormones = heart.get_hormones()
            
            # Determine Instinctual Urge
            instinct_urge = network_action['type']
            if hormones['cortisol'] > 0.8:
                instinct_urge = 'PANIC' # The Lizard Brain takes over
            
            # 4. CORTEX (System 2 "Reason")
            # Deliberate: Should we follow the urge or override?
            final_decision = cortex.deliberate(world_state, instinct_urge, hormones)
            
            # 5. EXECUTION
            real_action = network_action # Default
            
            if final_decision == 'PANIC':
                # Force random chaotic action
                real_action = {'type': 'NOISE', 'x': np.random.randint(0, 8), 'y': np.random.randint(0, 8), 'value': 1.0}
            elif final_decision == 'STAY_CALM':
                # Force network action (suppress panic)
                real_action = network_action
            elif final_decision == 'FOCUS_ORDER':
                # Force ordering action
                real_action = {'type': 'SYMMETRIZE', 'axis': 0}
            else:
                # Normal execution
                real_action = network_action
                # If instinct was NOISE but decision was NOISE, it passes through here
            
            # Apply Action
            world.apply_action(real_action)
            
            # 6. LEARNING (The "Feeling")
            # Loss = Cortisol (Pain) - [Dopamine (Pleasure) + Serotonin (Meaning)]
            # We want to Minimize Loss => Minimize Pain, Maximize Pleasure/Meaning
            
            # Calculate hormone deltas for memory
            hormone_delta = {
                'dopamine': hormones['dopamine'] - prev_hormones['dopamine'],
                'serotonin': hormones['serotonin'] - prev_hormones['serotonin'],
                'cortisol': hormones['cortisol'] - prev_hormones['cortisol']
            }
            
            # Store Memory
            cortex.store_memory(world_state, final_decision, hormone_delta)
            
            # Backpropagate
            # We construct a loss function that aligns with the hormone objective
            # Loss = (Cortisol * 2) - (Dopamine + Serotonin)
            # This is a reinforcement learning signal.
            # Since we are using differentiable components (Mind/Body), we can try to guide them.
            # However, hormones are calculated from non-differentiable world state.
            # So we use the "Hormone State" as the target.
            
            # Simple Proxy Loss:
            # If Cortisol is high, penalize current state/action.
            # If Dopamine/Serotonin is high, reinforce.
            
            reward = (hormones['dopamine'] + hormones['serotonin']) - (hormones['cortisol'] * 2.0)
            
            # Policy Gradient-like loss: -log(prob) * reward
            # But we have a continuous action space mixed with discrete.
            # Let's use a simpler proxy:
            # Minimize Energy (Physical) + Maximize Consistency (Mental) weighted by Hormones.
            
            # Actually, let's trust the user's design: "Loss = Cortisol - ..."
            # We treat the hormone values as the loss magnitude.
            
            total_loss = (hormones['cortisol'] * 2.0) - (hormones['dopamine'] + hormones['serotonin'])
            
            # Add consistency loss (always want truth)
            total_loss += (1.0 - consistency)
            
            # We need gradients. 
            # Since world interaction is non-differentiable, we use the "REINFORCE" idea or simply
            # attach the loss to the output logits if we had a target.
            # Here, we simply backprop the consistency and add a "regret" term for the action?
            # For simplicity in this version, we'll focus on the Consistency Loss + EWC, 
            # and let the "Evolutionary" selection (or just the hormone dynamics driving the loop) handle the rest.
            # Wait, if we don't backprop the hormone signal, the network won't learn to optimize hormones.
            
            # Fix: Use the Reward to scale the gradients?
            # Let's use a simple trick: Loss = Consistency + (Energy * Cortisol)
            # If Cortisol is high, Energy minimization becomes critical.
            loss_tensor = (1.0 - consistency) + (torch.tensor(world_energy) * hormones['cortisol'])
            
            optimizer.zero_grad()
            loss_tensor.backward()
            optimizer.step()
            
            # Update history
            prev_hormones = hormones.copy()
            history['energy'].append(world_energy)
            history['cortisol'].append(hormones['cortisol'])
            history['willpower'].append(cortex.willpower)
            
            # Logging
            if step % 50 == 0:
                print(f"S{step:03d} | E:{world_energy:.3f} | "
                      f"Cort:{hormones['cortisol']:.2f} Dopa:{hormones['dopamine']:.2f} Sero:{hormones['serotonin']:.2f} | "
                      f"Will:{cortex.willpower:.2f} | Act:{real_action['type']:10s} | Dec:{final_decision}")
                
            # Nirvana Check
            if hormones['serotonin'] > 0.9 and hormones['cortisol'] < 0.1:
                print(f"\n[NIRVANA] The Agent has found Peace at step {step}.")
                break
                
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            break
            
    print("="*80)
    print("Final Stats:")
    print(f"Energy: {history['energy'][-1]:.4f}")
    print(f"Avg Cortisol: {np.mean(history['cortisol']):.2f}")
    print(f"Avg Willpower: {np.mean(history['willpower']):.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
