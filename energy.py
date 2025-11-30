import numpy as np

class NeuroChemicalEngine:
    """
    The Heart.
    Manages the emotional state and motivation of the agent via Dopamine and Serotonin.
    """
    def __init__(self):
        self.dopamine = 0.5  # Drive / Curiosity
        self.serotonin = 0.5 # Peace / Stability
        self.prev_energy = float('inf')
        
        # History for variance calculation
        self.energy_history = []

    def update(self, current_energy, consistency_score):
        """
        Update hormone levels based on energy (error) and consistency (truth).
        
        Args:
            current_energy (float): The current error/loss of the system.
            consistency_score (float): How close the mind is to the Axiom (0-1).
        """
        # 1. Dopamine Dynamics (Reward Prediction Error)
        # Spike when energy drops significantly (Improvement)
        energy_delta = self.prev_energy - current_energy
        if energy_delta > 0:
            self.dopamine += energy_delta * 2.0  # Boost drive
        else:
            self.dopamine *= 0.95 # Decay if no progress
        
        self.dopamine = np.clip(self.dopamine, 0.0, 1.0)
        self.prev_energy = current_energy

        # 2. Serotonin Dynamics (Homeostasis & Truth)
        # Rise when consistent with Truth and Energy is low/stable
        is_stable = False
        if len(self.energy_history) > 5:
            variance = np.var(self.energy_history[-5:])
            if variance < 0.01:
                is_stable = True
        
        if is_stable and consistency_score > 0.8:
            self.serotonin += 0.1
        else:
            self.serotonin *= 0.98 # Decay if chaotic
            
        self.serotonin = np.clip(self.serotonin, 0.0, 1.0)
        self.energy_history.append(current_energy)

    def get_state(self):
        """
        Returns the dominant state: 'CHAOS' (Dopamine driven) or 'ORDER' (Serotonin driven).
        """
        if self.dopamine > self.serotonin:
            return "CHAOS"
        else:
            return "ORDER"

    def get_hormones(self):
        return self.dopamine, self.serotonin
