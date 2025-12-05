import torch
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
from environment_manager import EnvironmentManager
from cortex import PrefrontalCortex
from meta_cognition import MetaLearner
import torch.optim as optim

def check_knowledge_absorption():
    print(">>> KNOWLEDGE ABSORPTION DIAGNOSTIC <<<")

    # Init Systems
    world = World(size=8)
    vision = VisionSystem()
    heart = NeuroChemicalEngine()
    cortex = PrefrontalCortex(memory_capacity=50) # System 2

    LATENT_DIM = 8
    v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
    mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=v_truth)
    body = ActionDecoder(latent_dim=LATENT_DIM)
    soul = IntrinsicAutomata(mind) # System 4 (Long-term)

    meta_learner = MetaLearner(input_dim=7, hidden_dim=64) # System 3 (Meta)

    optimizer = optim.Adam(list(mind.parameters()) + list(body.parameters()), lr=0.01)

    # Trackers
    initial_meta_weights = [p.clone().detach() for p in meta_learner.parameters()]

    print("\n[1] Starting Simulation Loop (50 Steps)...")

    meta_hidden = None
    prev_hormones = {'dopamine': 0.5, 'serotonin': 0.5, 'cortisol': 0.0}
    prev_energy = 100.0
    act_map = {'DRAW': 0, 'NOISE': 1, 'CLEAR': 2, 'SYMMETRIZE': 3}
    grad_norm = 0.0

    for step in range(1, 51):
        # ... (Standard Cycle) ...
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
        if final_decision == 'PANIC': real_action = {'type': 'NOISE', 'x': 0, 'y': 0}
        elif final_decision == 'FOCUS_ORDER': real_action = {'type': 'SYMMETRIZE', 'axis': 0}

        world.apply_action(real_action)

        # --- KNOWLEDGE CHECK 1: EPISODIC MEMORY (Cortex) ---
        hormone_delta = {
            'dopamine': hormones['dopamine'] - prev_hormones['dopamine'],
            'serotonin': hormones['serotonin'] - prev_hormones['serotonin'],
            'cortisol': hormones['cortisol'] - prev_hormones['cortisol']
        }
        cortex.store_memory(world_state, final_decision, hormone_delta)

        # --- KNOWLEDGE CHECK 2: META-LEARNING (System 3) ---
        act_val = act_map.get(real_action['type'], 0)
        energy_delta = prev_energy - world_energy
        current_meta_reward = (hormones['dopamine'] + hormones['serotonin']) - hormones['cortisol'] + energy_delta * 2.0

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
        if step % 10 == 0:
            meta_learner.update()

        current_lr = 0.01 * meta_lr.item()
        for pg in optimizer.param_groups: pg['lr'] = current_lr

        probs = torch.softmax(action_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum()
        loss = (1.0 - consistency) + (torch.tensor(world_energy) * hormones['cortisol'] * meta_cort) - (entropy * meta_ent)

        optimizer.zero_grad()
        loss.backward()

        total_norm = 0.0
        for p in list(mind.parameters()) + list(body.parameters()):
            if p.grad is not None: total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5

        optimizer.step()

        # --- KNOWLEDGE CHECK 3: CRYSTALLIZATION (Soul) ---
        soul.update_state((hormones['dopamine'], hormones['serotonin']))
        # Force check EWC loss calculation (usually happens on Nirvana)
        if step == 25:
             # Simulate Nirvana trigger
             soul.crystallize(mind)

        prev_hormones = hormones.copy()
        prev_energy = world_energy

    print("\n[2] Analysis Results:")

    # 1. Check Cortex Memory
    mem_count = len(cortex.memory)
    print(f"  A. Episodic Memory: {mem_count}/50 slots filled.")
    if mem_count > 0:
        print("     [PASS] Cortex is storing experiences.")
    else:
        print("     [FAIL] Cortex memory is empty.")

    # 2. Check Meta-Learner Plasticity
    changed = False
    diff_sum = 0.0
    for p_new, p_old in zip(meta_learner.parameters(), initial_meta_weights):
        diff = (p_new - p_old).abs().sum().item()
        diff_sum += diff
        if diff > 0:
            changed = True

    print(f"  B. Meta-Knowledge: Weight Delta = {diff_sum:.6f}")
    if changed:
        print("     [PASS] Meta-Learner is updating its synaptic weights.")
    else:
        print("     [FAIL] Meta-Learner is static (Learning frozen).")

    # 3. Check Soul Crystallization
    print(f"  C. Semantic Knowledge (Soul): Crystallized = {soul.crystallized}")
    if soul.crystallized:
        print(f"     [PASS] Soul has crystallized knowledge (Locked {len(soul.optpar)} params).")
    else:
        print("     [FAIL] No knowledge crystallization occurred.")

    print("\n>>> DIAGNOSTIC COMPLETE <<<")

if __name__ == "__main__":
    check_knowledge_absorption()
