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

MAX_GENERATIONS = 100000 # Effectively infinite
STEPS_PER_GEN = 1000
META_BRAIN_FILE = "meta_brain.pth"

def run_generation(gen_id, meta_learner):
    # randomness per generation
    seed = int(time.time()) + gen_id * 997
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n[GEN {gen_id}] Initializing Life... (Seed {seed})")

    # 1. Init Base Agent (Tabula Rasa)
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

    # Tracking
    meta_hidden = None
    prev_hormones = {'dopamine': 0.5, 'serotonin': 0.5, 'cortisol': 0.0}
    prev_energy = 100.0
    act_map = {'DRAW': 0, 'NOISE': 1, 'CLEAR': 2, 'SYMMETRIZE': 3}
    grad_norm = 0.0

    best_energy = float('inf')
    nirvana_step = None

    start_time = time.time()

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

            # Symmetry
            h_sym = 1.0 - np.abs(world_state - np.fliplr(world_state)).mean()
            v_sym = 1.0 - np.abs(world_state - np.flipud(world_state)).mean()
            symmetry = (h_sym + v_sym) / 2

            prediction_error = 1.0 - consistency.item()
            heart.update(world_energy, consistency.item(), density, symmetry, prediction_error)
            hormones = heart.get_hormones()

            # Cortex Decision
            instinct_urge = network_action['type']
            if hormones['cortisol'] > 0.8: instinct_urge = 'PANIC'
            final_decision = cortex.deliberate(world_state, instinct_urge, hormones)

            real_action = network_action
            if final_decision == 'PANIC':
                real_action = {'type': 'NOISE', 'x': np.random.randint(0,8), 'y': np.random.randint(0,8)}
            elif final_decision == 'FOCUS_ORDER':
                real_action = {'type': 'SYMMETRIZE', 'axis': 0}

            world.apply_action(real_action)

            # --- META-LEARNING (System 3) ---
            act_val = act_map.get(real_action['type'], 0)
            energy_delta = prev_energy - world_energy

            current_meta_reward = (hormones['dopamine'] + hormones['serotonin']) - hormones['cortisol']
            current_meta_reward += energy_delta * 2.0

            # INVARIANT INPUTS (Dim 7)
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

            meta_learner.store_step(meta_log_prob, current_meta_reward)
            if step % 20 == 0:
                meta_learner.update()

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
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

            optimizer.step()

            # State Update
            prev_hormones = hormones.copy()
            prev_energy = world_energy
            best_energy = min(best_energy, world_energy)

            # Nirvana?
            if hormones['serotonin'] > 0.95 and hormones['cortisol'] < 0.05 and world_energy < 0.1:
                nirvana_step = step
                break

        except Exception as e:
            print(f"[ERROR] Step {step}: {e}")
            break

    steps_taken = nirvana_step if nirvana_step else STEPS_PER_GEN
    return steps_taken, best_energy

def main():
    print(">>> AUTONOMOUS EVOLUTION ENGINE STARTED <<<")
    print(f"PID: {os.getpid()}")

    # Load or Init Meta-Brain
    meta_learner = MetaLearner(input_dim=7, hidden_dim=64)
    if os.path.exists(META_BRAIN_FILE):
        try:
            meta_learner.load_state_dict(torch.load(META_BRAIN_FILE))
            print("Loaded existing Meta-Brain.")
        except:
            print("Could not load Meta-Brain, starting fresh.")

    gen = 1
    while gen <= MAX_GENERATIONS:
        steps, energy = run_generation(gen, meta_learner)

        # Save Progress
        torch.save(meta_learner.state_dict(), META_BRAIN_FILE)

        print(f"[RESULT] Gen {gen}: Steps={steps} | Best Energy={energy:.4f}")
        print(f"Meta-Brain saved to {META_BRAIN_FILE}")

        # Flush stdout
        sys.stdout.flush()
        gen += 1

if __name__ == "__main__":
    main()
