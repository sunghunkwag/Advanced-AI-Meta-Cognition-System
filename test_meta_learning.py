import torch
import numpy as np
import torch.optim as optim

# Importing classes
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

def run_simulation_instance(use_meta=True, steps=500):
    # Init
    world = World(size=8)
    vision = VisionSystem()
    heart = NeuroChemicalEngine()
    cortex = PrefrontalCortex()

    # Meta-Learner (Input Dim 12)
    meta_learner = MetaLearner(input_dim=12, hidden_dim=64)
    meta_hidden = None

    # Mind/Body
    LATENT_DIM = 8
    v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
    mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=v_truth)
    body = ActionDecoder(latent_dim=LATENT_DIM)

    optimizer = optim.Adam(list(mind.parameters()) + list(body.parameters()), lr=0.01)

    prev_hormones = {'dopamine': 0.5, 'serotonin': 0.5, 'cortisol': 0.0}
    prev_energy = 100.0
    act_map = {'DRAW': 0, 'NOISE': 1, 'CLEAR': 2, 'SYMMETRIZE': 3}
    grad_norm = 0.0

    final_energy = 0.0

    for step in range(steps):
        # 1. Perception
        world_state = world.get_state()
        nodes, adj = vision.perceive(world_state)

        # 2. Mind
        z = mind(nodes, adj)
        consistency = mind.check_consistency(z)

        # 3. Action
        action_logits, params = body(z)
        network_action = body.decode_action(action_logits, params)

        # 4. Hormones
        world_energy = world.calculate_energy()
        density = np.sum(world_state > 0.1) / world_state.size
        h_sym = 1.0 - np.abs(world_state - np.fliplr(world_state)).mean()
        v_sym = 1.0 - np.abs(world_state - np.flipud(world_state)).mean()
        symmetry = (h_sym + v_sym) / 2
        prediction_error = 1.0 - consistency.item()

        heart.update(world_energy, consistency.item(), density, symmetry, prediction_error)
        hormones = heart.get_hormones()

        # 5. Decision
        real_action = network_action
        world.apply_action(real_action)

        # 6. Meta-Learning
        meta_cort = torch.tensor(1.0)
        meta_ent = torch.tensor(0.01)

        if use_meta:
            act_val = act_map.get(real_action['type'], 0)
            energy_delta = prev_energy - world_energy
            current_meta_reward = (hormones['dopamine'] + hormones['serotonin']) - hormones['cortisol']
            current_meta_reward += energy_delta * 2.0

            meta_input = torch.cat([
                z.detach().float(),
                torch.tensor([[float(act_val)]]).float(),
                torch.tensor([[current_meta_reward]]).float(),
                torch.tensor([[float(grad_norm)]]).float(),
                torch.tensor([[float(energy_delta)]]).float()
            ], dim=1).unsqueeze(0)

            (meta_lr, meta_cort, meta_ent), meta_log_prob, meta_hidden = meta_learner(meta_input, meta_hidden)
            meta_hidden = (meta_hidden[0].detach(), meta_hidden[1].detach())

            meta_learner.store_step(meta_log_prob, current_meta_reward)
            if step % 20 == 0 and step > 0:
                meta_learner.update()

            current_lr = 0.01 * meta_lr.item()
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # 7. Learning
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
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5

        optimizer.step()
        prev_energy = world_energy
        final_energy = world_energy

    return final_energy

if __name__ == "__main__":
    print("Running Comparative Verification...")

    # Run Baseline (5 runs)
    baseline_scores = []
    print("\nBaseline (Fixed LR=0.01):")
    for i in range(5):
        score = run_simulation_instance(use_meta=False, steps=200)
        baseline_scores.append(score)
        print(f"Run {i+1}: Final Energy = {score:.4f}")

    # Run Meta (5 runs)
    meta_scores = []
    print("\nMeta-Learner (Adaptive):")
    for i in range(5):
        score = run_simulation_instance(use_meta=True, steps=200)
        meta_scores.append(score)
        print(f"Run {i+1}: Final Energy = {score:.4f}")

    avg_base = np.mean(baseline_scores)
    avg_meta = np.mean(meta_scores)

    print("\n" + "="*40)
    print(f"Average Baseline Energy: {avg_base:.4f}")
    print(f"Average Meta Energy:     {avg_meta:.4f}")
    print(f"Improvement:             {avg_base - avg_meta:.4f}")
    print("="*40)
