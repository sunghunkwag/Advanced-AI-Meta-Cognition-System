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
from meta_cognition import MetaLearner

def main():
    print("[INIT] Advanced AI: Dual-Process Architecture + Meta-Learning")
    print("="*80)
    
    # Initialize Systems
    env_manager = EnvironmentManager()
    world = env_manager.create_environment("tiny") # 8x8
    
    # System 1: The Heart (Hormones)
    heart = NeuroChemicalEngine()
    
    # System 2: The Cortex (Reason)
    cortex = PrefrontalCortex()
    
    # System 3: Meta-Learner (Self-Improvement)
    # Input: Latent(8) + Action(1) + Reward(1) = 10
    meta_learner = MetaLearner(input_dim=10, hidden_dim=32)
    meta_hidden = None
    meta_update_interval = 20

    # Body & Mind
    vision = VisionSystem()
    LATENT_DIM = 8
    v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
    mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=v_truth)
    body = ActionDecoder(latent_dim=LATENT_DIM)
    soul = IntrinsicAutomata(mind)
    
    base_lr = 0.01
    optimizer = optim.Adam(list(mind.parameters()) + list(body.parameters()), lr=base_lr)
    
    # Tracking
    history = {
        'energy': [], 'dopamine': [], 'serotonin': [], 'cortisol': [],
        'willpower': [], 'actions': [], 'meta_lr': [], 'meta_cort': []
    }
    
    print("[OK] Systems Online. Starting Life Cycle...")
    print("="*80)
    
    prev_hormones = {'dopamine': 0.5, 'serotonin': 0.5, 'cortisol': 0.0}
    act_map = {'DRAW': 0, 'NOISE': 1, 'CLEAR': 2, 'SYMMETRIZE': 3}
    
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
            world_energy = world.calculate_energy()
            density = np.sum(world_state > 0.1) / world_state.size
            
            h_sym = 1.0 - np.abs(world_state - np.fliplr(world_state)).mean()
            v_sym = 1.0 - np.abs(world_state - np.flipud(world_state)).mean()
            symmetry = (h_sym + v_sym) / 2
            
            prediction_error = 1.0 - consistency.item()
            
            heart.update(world_energy, consistency.item(), density, symmetry, prediction_error)
            hormones = heart.get_hormones()
            
            instinct_urge = network_action['type']
            if hormones['cortisol'] > 0.8:
                instinct_urge = 'PANIC'
            
            # 4. CORTEX (System 2 "Reason")
            final_decision = cortex.deliberate(world_state, instinct_urge, hormones)
            
            # 5. EXECUTION
            real_action = network_action
            if final_decision == 'PANIC':
                real_action = {'type': 'NOISE', 'x': np.random.randint(0, 8), 'y': np.random.randint(0, 8), 'value': 1.0}
            elif final_decision == 'STAY_CALM':
                real_action = network_action
            elif final_decision == 'FOCUS_ORDER':
                real_action = {'type': 'SYMMETRIZE', 'axis': 0}
            else:
                real_action = network_action
            
            world.apply_action(real_action)
            
            # --- META-COGNITION (System 3) ---
            # Prepare Input for Meta-Learner
            act_val = act_map.get(real_action['type'], 0)
            current_meta_reward = (hormones['dopamine'] + hormones['serotonin']) - hormones['cortisol']
            
            meta_input = torch.cat([
                z.detach().float(), # (1, 8)
                torch.tensor([[float(act_val)]]).float(), # (1, 1)
                torch.tensor([[current_meta_reward]]).float() # (1, 1)
            ], dim=1).unsqueeze(0) # (1, 1, 10)
            
            # Predict Learning Hyperparameters
            (meta_lr, meta_cort, meta_ent), meta_log_prob, meta_hidden = meta_learner(meta_input, meta_hidden)
            
            # Detach hidden state to prevent infinite backprop through time
            meta_hidden = (meta_hidden[0].detach(), meta_hidden[1].detach())
            
            # Store for Meta-Update
            meta_learner.store_step(meta_log_prob, current_meta_reward)
            
            if step % meta_update_interval == 0:
                meta_learner.update()
            
            # 6. LEARNING
            # Apply dynamic learning rate
            current_lr = base_lr * meta_lr.item()
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Calculate Entropy of action distribution
            probs = torch.softmax(action_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum()
            
            # Dynamic Loss Function
            # We want to minimize (Energy * Cortisol_Sensitivity) and maximize (Consistency)
            # We also want to maximize Entropy based on Meta-Learner's curiosity suggestion
            
            loss_tensor = (1.0 - consistency) + \
                          (torch.tensor(world_energy) * hormones['cortisol'] * meta_cort) - \
                          (entropy * meta_ent)
            
            optimizer.zero_grad()
            loss_tensor.backward()
            optimizer.step()
            
            # Update history
            hormone_delta = {
                'dopamine': hormones['dopamine'] - prev_hormones['dopamine'],
                'serotonin': hormones['serotonin'] - prev_hormones['serotonin'],
                'cortisol': hormones['cortisol'] - prev_hormones['cortisol']
            }
            cortex.store_memory(world_state, final_decision, hormone_delta)

            prev_hormones = hormones.copy()
            history['energy'].append(world_energy)
            history['cortisol'].append(hormones['cortisol'])
            history['meta_lr'].append(meta_lr.item())
            history['meta_cort'].append(meta_cort.item())
            
            # Logging
            if step % 50 == 0:
                print(f"S{step:03d} | E:{world_energy:.3f} | "
                      f"Cort:{hormones['cortisol']:.2f} Dopa:{hormones['dopamine']:.2f} | "
                      f"Meta[LR:{meta_lr.item():.2f} C:{meta_cort.item():.2f} Ent:{meta_ent.item():.2f}] | "
                      f"Act:{real_action['type']}")
                
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
    print(f"Avg Meta LR Scale: {np.mean(history['meta_lr']):.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
