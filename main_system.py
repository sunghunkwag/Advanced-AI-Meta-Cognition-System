import torch
import torch.optim as optim
import time
import numpy as np

# Import Core Modules
from vision import VisionSystem
from manifold import GraphAttentionManifold
from energy import NeuroChemicalEngine, JEPA_Predictor
from action_decoder import ActionDecoder
from automata import IntrinsicAutomata
from world import World
from soul import get_soul_vectors
from meta_cognition import MetaCognitiveController
from planner import TreeSearchPlanner
from imagination import LatentWorldModel

def main():
    print("🚀 Initializing Advanced AI System (Awakened)...")
    
    # 1. Initialize Modules
    world = World(size=16)
    vision = VisionSystem()
    
    # Soul Injection
    # V3.5: Get Identity, Truth, and Reject vectors
    # We use dim=8 to match the Latent Dimension of the Mind
    LATENT_DIM = 8
    v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
    
    # Mind (GAT): Input features=3 (y, x, val), Hidden=16, Output=Latent Dim (8)
    # Inject Truth Vector for Axiom Consistency
    mind = GraphAttentionManifold(nfeat=3, nhid=16, nclass=LATENT_DIM, truth_vector=v_truth)
    
    # Body: Latent Dim (8) -> Action
    body = ActionDecoder(latent_dim=LATENT_DIM)
    
    # Heart (Neuro-Chemical Engine)
    heart = NeuroChemicalEngine()
    
    # Heart (JEPA Predictor for Imagination)
    # State=8, Action=4 (Logits) + 4 (Params) = 8? 
    # Or just Action Logits? Planner uses Action ID.
    # Let's assume Action Dim = 4 (Logits) for simplicity in Imagination.
    # But Body outputs 4 logits + 4 params.
    # Let's stick to Action Dim = 4 for the JEPA input (representing intention).
    jepa = JEPA_Predictor(state_dim=LATENT_DIM, action_dim=4)
    
    # Soul
    soul = IntrinsicAutomata(mind) # Soul watches the Mind
    
    # System 2 Components
    meta_controller = MetaCognitiveController()
    imagination = LatentWorldModel(predictor=jepa, v_truth=v_truth)
    planner = TreeSearchPlanner(world_model=imagination, action_decoder=body, depth=2, num_actions=4)
    
    # Optimizer (The biological learning process)
    # We optimize Mind, Body, and JEPA
    optimizer = optim.Adam(list(mind.parameters()) + list(body.parameters()) + list(jepa.parameters()), lr=0.01)

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

        # E. Meta-Cognition (System 1 vs System 2)
        action_logits, params = body(z)
        
        # Calculate Entropy of action distribution
        probs = torch.softmax(action_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        
        meta_mode, reason = meta_controller.decide_mode(entropy)
        
        if meta_mode == "SYSTEM_2":
            # Think before acting
            print(f"🛑 [System 2 Triggered] {reason}")
            # Plan: Find best action ID
            best_action_id, predicted_energy, plan_trace = planner.plan(z)
            
            # Override Body's intuition with Planner's decision
            # We create a new logits vector favoring the best action
            # Or just force the action type in the decoder.
            # Let's hack the logits to force the decision.
            action_logits = torch.zeros_like(action_logits)
            action_logits[0, best_action_id] = 10.0 # Strong confidence
            
            print(f"   -> 🧠 Plan: {plan_trace}")
        
        # F. Body (Action Execution)
        action = body.decode_action(action_logits, params)
        
        # G. Act on World
        world.apply_action(action)
        
        # H. Learning (Backprop)
        # Loss = Energy + (1 - Consistency) + EWC
        # Also train JEPA? Yes, self-supervised.
        # JEPA Loss: ||JEPA(z, a) - z_next||
        # But we don't have z_next yet (it's the next step's z).
        # Simplified: We just train on the main objective for now.
        
        loss = torch.tensor(world_energy, dtype=torch.float32) + (1.0 - consistency) + ewc_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update Meta-Cognition History
        meta_controller.update_energy(world_energy)

        # Logging
        print(f"Step {step:02d} | Mode: {state_mode} | Dopa: {dopamine:.2f} Sero: {serotonin:.2f} | Energy: {world_energy:.4f} | Action: {action['type']}")
        
        if soul.is_crystallized():
            print(f"       -> 🧘 Nirvana Reached. Mind is Still.")
            break
            
        time.sleep(0.05)

    print("-" * 50)
    print("🏁 Simulation Complete.")

if __name__ == "__main__":
    main()
