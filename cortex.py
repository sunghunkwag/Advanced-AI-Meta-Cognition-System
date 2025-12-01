import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

class PrefrontalCortex:
    """
    System 2: The Rational Mind.
    Manages Episodic Memory, Willpower, and Inhibition.
    """
    def __init__(self, memory_capacity=1000):
        self.memory = []  # List of (state_vector, action, outcome_hormones)
        self.capacity = memory_capacity
        self.willpower = 1.0  # Starts full
        self.willpower_regen_rate = 0.05  # Increased from 0.01 (Faster recovery)
        
    def recall(self, current_state: np.ndarray) -> Optional[Dict]:
        """
        Retrieve the most similar past experience.
        
        Args:
            current_state: Current world state (flattened or feature vector)
            
        Returns:
            Dict containing 'action', 'outcome' (hormone change), 'similarity'
        """
        if not self.memory:
            return None
            
        # Simple similarity search (Euclidean distance on flattened state)
        # In a real brain, this is associative memory.
        best_match = None
        min_dist = float('inf')
        
        flat_state = current_state.flatten()
        
        # Optimization: Only check recent 50 memories for speed
        # (Heuristic: Recent context matters most)
        search_space = self.memory[-50:] if len(self.memory) > 50 else self.memory
        
        for mem in search_space:
            mem_state = mem['state'].flatten()
            dist = np.linalg.norm(flat_state - mem_state)
            
            if dist < min_dist:
                min_dist = dist
                best_match = mem
                
        if best_match and min_dist < 5.0: # Threshold for "similarity"
            return {
                'action': best_match['action'],
                'outcome': best_match['outcome'], # Hormone delta
                'similarity': 1.0 / (1.0 + min_dist)
            }
        return None

    def deliberate(self, current_state: np.ndarray, instinct_action: str, hormones: Dict) -> str:
        """
        Decide whether to follow instinct or override it based on memory and willpower.
        
        Args:
            current_state: Current world state
            instinct_action: Action proposed by System 1 (NeuroChemicalEngine)
            hormones: Current hormone levels
            
        Returns:
            Final Action (str)
        """
        # Regenerate willpower slightly
        self.willpower = min(1.0, self.willpower + self.willpower_regen_rate)
        
        memory = self.recall(current_state)
        
        # === CONFLICT 1: PANIC CONTROL (Cortisol) ===
        if hormones['cortisol'] > 0.6: # Lowered threshold slightly to catch stress earlier
            # Instinct wants to PANIC (Random Action / NOISE)
            # Reason checks: "Did panic help last time?"
            
            if memory:
                # If previous panic led to bad outcome (more cortisol/less dopamine), inhibit it.
                # Outcome metric: Dopamine + Serotonin - Cortisol
                past_score = memory['outcome']['dopamine'] + memory['outcome']['serotonin'] - memory['outcome']['cortisol']
                
                if past_score < 0: # It was bad!
                    # Try to inhibit
                    if self.willpower > hormones['cortisol'] * 0.5: # Cheaper to inhibit (0.5 multiplier)
                        self.willpower -= 0.05 # Reduced cost from 0.1
                        print(f"[PFC] Inhibiting PANIC! Willpower: {self.willpower:.2f}")
                        return "STAY_CALM" # Special signal to maintain previous strategy
                    else:
                        print("[PFC] Willpower depleted... PANIC ALLOWED.")
                        return instinct_action
            
            # If no memory, we might default to observing or letting instinct run.
            # But high cortisol usually demands action.
            # Default: Try to stay calm if we have willpower, even without memory (Blind Faith)
            if self.willpower > 0.5:
                 self.willpower -= 0.05
                 return "STAY_CALM"
            
            return instinct_action

        # === CONFLICT 2: IMPULSE CONTROL (Dopamine vs Serotonin) ===
        # Instinct wants NOISE (Novelty/Dopamine) but Serotonin is low (No Meaning)
        if instinct_action == 'NOISE' and hormones['serotonin'] < 0.3:
            # "I'm bored, let's mess it up!" vs "We haven't built anything yet!"
            
            if self.willpower > 0.2:
                self.willpower -= 0.02 # Reduced cost from 0.05
                print(f"[PFC] Inhibiting DISTRACTION! Focusing on Order. Willpower: {self.willpower:.2f}")
                return "FOCUS_ORDER" # Signal to prioritize SYMMETRIZE/DRAW
                
        return instinct_action

    def store_memory(self, state: np.ndarray, action: str, hormone_delta: Dict):
        """
        Store the experience for future reference.
        """
        memory_item = {
            'state': state.copy(),
            'action': action,
            'outcome': hormone_delta
        }
        self.memory.append(memory_item)
        
        if len(self.memory) > self.capacity:
            self.memory.pop(0) # Forget oldest
