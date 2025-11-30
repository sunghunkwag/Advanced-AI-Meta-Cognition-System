import torch
import torch.optim as optim
import numpy as np

# Import Core Modules  
from vision import VisionSystem
from manifold import GraphAttentionManifold
from energy import NeuroChemicalEngine
from action_decoder import ActionDecoder
from world import World
from soul import get_soul_vectors

print("=" * 60)
print("DEBUG: Minimal Learning Test")
print("=" * 60)

# Initialize
world = World(size=16)
vision = VisionSystem()

LATENT_DIM = 8
v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=v_truth)
body = ActionDecoder(latent_dim=LATENT_DIM)
heart = NeuroChemicalEngine()

optimizer = optim.Adam(list(mind.parameters()) + list(body.parameters()), lr=0.01)

print("\n[INITIAL STATE]")
initial_energy = world.calculate_energy()
print(f"Initial Energy: {initial_energy:.4f}")
print(f"Grid sum: {world.grid.sum():.4f}")

# Run 10 steps with forced diversity
for step in range(1, 11):
    # Perceive
    nodes, adj = vision.perceive(world.get_state())
    z = mind(nodes, adj)
    
    # Action with forced exploration
    action_logits, params = body(z)
    
    # Force different actions each step for testing
    forced_action_idx = step % 4
    action_logits = torch.zeros_like(action_logits)
    action_logits[0, forced_action_idx] = 10.0
    
    action = body.decode_action(action_logits, params)
    
    # Apply
    world.apply_action(action)
    new_energy = world.calculate_energy()
    consistency = mind.check_consistency(z)
    
    # Learn
    loss = torch.tensor(new_energy, dtype=torch.float32) + (1.0 - consistency)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    heart.update(new_energy, consistency.item())
    dopa, sero = heart.get_hormones()
    
    print(f"\nStep {step:02d}: {action['type']:12s} | Energy: {new_energy:.4f} | Loss: {loss.item():.4f} | Dopa: {dopa:.2f} Sero: {sero:.2f}")
    print(f"        Grid sum: {world.grid.sum():.4f} | Grid mean: {world.grid.mean():.4f}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
