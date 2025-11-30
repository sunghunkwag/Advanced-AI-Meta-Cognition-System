import torch
import torch.optim as optim
import time
import numpy as np

# Import Core Modules
from vision import VisionSystem
from manifold import GraphAttentionManifold
from energy import NeuroChemicalEngine
from action_decoder import ActionDecoder
from automata import IntrinsicAutomata
from world import World

def main():
    print("🚀 Initializing Advanced AI System...")
    
    # 1. Initialize Modules
    world = World(size=16)
    vision = VisionSystem()
    
    # Mind (GAT): Input features=3 (y, x, val), Hidden=16, Output=Latent Dim (8)
    mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=8)
    
    # Body: Latent Dim (8) -> Action
    body = ActionDecoder(latent_dim=8)
    
    # Heart
    heart = NeuroChemicalEngine()
    
    # Soul
    soul = IntrinsicAutomata(mind) # Soul watches the Mind
    
    # Optimizer (The biological learning process)
    optimizer = optim.Adam(list(mind.parameters()) + list(body.parameters()), lr=0.01)

    print("✅ System Online. Beginning Life Cycle.")
    print("-" * 50)

    # Simulation Loop
    for step in range(1, 51): # 50 Steps of life
        # A. Perception
        world_state = world.get_state()
        nodes, adj = vision.perceive(world_state)
        
        # B. Mind (Reasoning)
        z = mind(nodes, adj)
        consistency = mind.check_consistency(z)
        
        # C. Heart (Feeling)
        # Calculate external energy (World state)
        world_energy = world.calculate_energy()
        heart.update(world_energy, consistency.item())
        dopamine, serotonin = heart.get_hormones()
        state_mode = heart.get_state()
        
        # D. Soul (Crystallization)
        soul.update_state((dopamine, serotonin))
        ewc_loss = soul.ewc_loss(mind)

        # E. Body (Action)
        action_logits, params = body(z)
        action = body.decode_action(action_logits, params)
        
        # F. Act on World
        world.apply_action(action)
        
        # G. Learning (Backprop)
        # Goal: Minimize World Energy (Symmetry) + Maximize Consistency
        # Loss = Energy + (1 - Consistency) + EWC
        loss = torch.tensor(world_energy, dtype=torch.float32) + (1.0 - consistency) + ewc_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        print(f"Step {step:02d} | Mode: {state_mode} | Dopa: {dopamine:.2f} Sero: {serotonin:.2f} | Energy: {world_energy:.4f} | Action: {action['type']}")
        
        if soul.is_crystallized():
            print(f"       -> 🧘 Nirvana Reached. Mind is Still.")
            break
            
        time.sleep(0.1)

    print("-" * 50)
    print("🏁 Simulation Complete.")

if __name__ == "__main__":
    main()
