import torch
import torch.optim as optim
import numpy as np
import time
import os
import sys

# Core Modules
from vision import VisionSystem
from manifold import GraphAttentionManifold
from energy import NeuroChemicalEngine
from action_decoder import ActionDecoder
from automata import IntrinsicAutomata
from world import World
from soul import get_soul_vectors
from environment_manager import EnvironmentManager
from cortex import PrefrontalCortex
from meta_cognition import MetaLearner

MAX_GENERATIONS = 100000
STEPS_PER_GEN = 1000
META_BRAIN_FILE = "meta_brain.pth"

def run_episode(meta_learner, episode_id):
    """
    Runs a single life-cycle episode.
    Returns: best_energy, final_state_dict (meta_learner)
    """
    # randomness
    seed = int(time.time()) + episode_id * 997
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Init Base Agent
    env_manager = EnvironmentManager()
    world = env_manager.create_environment("tiny")
    vision = VisionSystem()
    heart = NeuroChemicalEngine()
    cortex = PrefrontalCortex()

    LATENT_DIM = 8
    v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
    mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=v_truth)
    body = ActionDecoder(latent_dim=LATENT_DIM)

    optimizer = optim.Adam(list(mind.parameters()) + list(body.parameters()), lr=0.01)

    # Clone Meta-Learner for this timeline to allow exploration
    # (In standard PPO we use the same policy, but here we let them diverge slightly during the episode
    # if we wanted, but for now we keep it shared but update individually)
    # Actually, we want to accumulate gradients or just find the best trajectory.
    # Let's run with the *shared* meta_learner but *don't update it yet*.
    # We will store the trajectory reward.

    meta_hidden = None
    prev_hormones = {'dopamine': 0.5, 'serotonin': 0.5, 'cortisol': 0.0}
    prev_energy = 100.0
    act_map = {'DRAW': 0, 'NOISE': 1, 'CLEAR': 2, 'SYMMETRIZE': 3}
    grad_norm = 0.0
    best_energy = float('inf')

    # We need to temporarily store experiences for this episode
    episode_meta_log_probs = []
    episode_rewards = []

    for step in range(1, STEPS_PER_GEN + 1):
        try:
            # --- AGENT CYCLE ---
            world_state = world.get_state()
            nodes, adj = vision.perceive(world_state)
            z = mind(nodes, adj)
            consistency = mind.check_consistency(z)

            action_logits, params = body(z)
            network_action = body.decode_action(action_logits, params)

            world_energy = world.calculate_energy()
            density = np.sum(world_state > 0.1) / world_state.size
            h_sym = 1.0 - np.abs(world_state - np.fliplr(world_state)).mean()
            v_sym = 1.0 - np.abs(world_state - np.flipud(world_state)).mean()
            symmetry = (h_sym + v_sym) / 2

            prediction_error = 1.0 - consistency.item()
            heart.update(world_energy, consistency.item(), density, symmetry, prediction_error)
            hormones = heart.get_hormones()

            instinct_urge = network_action['type']
            if hormones['cortisol'] > 0.8: instinct_urge = 'PANIC'
            final_decision = cortex.deliberate(world_state, instinct_urge, hormones)

            real_action = network_action
            if final_decision == 'PANIC':
                real_action = {'type': 'NOISE', 'x': np.random.randint(0,8), 'y': np.random.randint(0,8)}
            elif final_decision == 'FOCUS_ORDER':
                real_action = {'type': 'SYMMETRIZE', 'axis': 0}

            world.apply_action(real_action)

            # --- META-LEARNING ---
            act_val = act_map.get(real_action['type'], 0)
            energy_delta = prev_energy - world_energy

            current_meta_reward = (hormones['dopamine'] + hormones['serotonin']) - hormones['cortisol']
            current_meta_reward += energy_delta * 2.0

            meta_input = torch.cat([
                torch.tensor([[float(consistency.item())]]).float(),
                torch.tensor([[float(symmetry)]]).float(),
                torch.tensor([[float(density)]]).float(),
                torch.tensor([[float(act_val)]]).float(),
                torch.tensor([[current_meta_reward]]).float(),
                torch.tensor([[float(grad_norm)]]).float(),
                torch.tensor([[float(energy_delta)]]).float()
            ], dim=1).unsqueeze(0)

            (meta_lr, meta_cort, meta_ent), meta_log_prob, meta_hidden = meta_learner(meta_input, meta_hidden)
            meta_hidden = (meta_hidden[0].detach(), meta_hidden[1].detach())

            # Store in local buffer
            episode_meta_log_probs.append(meta_log_prob)
            episode_rewards.append(current_meta_reward)

            # Apply Parameters
            current_lr = 0.01 * meta_lr.item()
            for pg in optimizer.param_groups: pg['lr'] = current_lr

            # Backprop
            probs = torch.softmax(action_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum()
            loss = (1.0 - consistency) + \
                   (torch.tensor(world_energy) * hormones['cortisol'] * meta_cort) - \
                   (entropy * meta_ent)

            optimizer.zero_grad()
            loss.backward()

            total_norm = 0.0
            for p in list(mind.parameters()) + list(body.parameters()):
                if p.grad is not None: total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

            optimizer.step()

            prev_hormones = hormones.copy()
            prev_energy = world_energy
            best_energy = min(best_energy, world_energy)

            if hormones['serotonin'] > 0.95 and hormones['cortisol'] < 0.05 and world_energy < 0.1:
                break

        except Exception as e:
            print(f"[ERROR] {e}")
            break

    return best_energy, episode_meta_log_probs, episode_rewards

def main():
    print(">>> ACCELERATED AUTONOMOUS EVOLUTION (ELITISM) <<<")

    meta_learner = MetaLearner(input_dim=7, hidden_dim=64)
    if os.path.exists(META_BRAIN_FILE):
        try:
            meta_learner.load_state_dict(torch.load(META_BRAIN_FILE))
            print("Loaded existing Meta-Brain.")
        except:
            print("Could not load Meta-Brain, starting fresh.")

    gen = 1
    POPULATION_SIZE = 3

    while gen <= MAX_GENERATIONS:
        print(f"\n[GEN {gen}] Spawning {POPULATION_SIZE} Timelines...")

        candidates = []

        # Sequential Population Execution
        for i in range(POPULATION_SIZE):
            energy, log_probs, rewards = run_episode(meta_learner, gen * 100 + i)
            candidates.append({
                'energy': energy,
                'log_probs': log_probs,
                'rewards': rewards
            })
            print(f"  Timeline {i+1}: Best Energy = {energy:.4f}")

        # Elitism Selection
        candidates.sort(key=lambda x: x['energy'])
        winner = candidates[0]
        print(f"[GEN {gen}] Winner Energy: {winner['energy']:.4f}")

        # Train Meta-Learner on the WINNER ONLY
        meta_learner.saved_log_probs = winner['log_probs']
        meta_learner.rewards = winner['rewards']
        meta_learner.update() # Now uses multi-epoch update

        torch.save(meta_learner.state_dict(), META_BRAIN_FILE)
        sys.stdout.flush()
        gen += 1

if __name__ == "__main__":
    main()
