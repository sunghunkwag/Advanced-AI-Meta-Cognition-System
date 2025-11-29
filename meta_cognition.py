"""
Meta-Cognition Module (V4.0 - System 2)

The "Manager" of the mind.
Decides WHEN to think (System 2) and WHEN to act (System 1).
Monitors internal states: Entropy (Uncertainty) and Energy Variance (Stability).
"""

import torch
import numpy as np
from collections import deque

class MetaCognitiveController:
    def __init__(self, entropy_threshold: float = 1.5, variance_threshold: float = 0.005):
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        self.energy_history = deque(maxlen=10)
        
    def update_energy(self, energy_value: float):
        self.energy_history.append(energy_value)
        
    def decide_mode(self, entropy: float) -> tuple[str, str]:
        """
        Decide whether to use System 1 (Intuition) or System 2 (Planning).
        
        Args:
            entropy: Current policy entropy (Uncertainty)
            
        Returns:
            mode: "SYSTEM_1" or "SYSTEM_2"
            reason: Explanation for the decision
        """
        # 1. Check Uncertainty (Entropy)
        # High entropy means the agent doesn't know what to do -> Need Planning
        if entropy > self.entropy_threshold:
            return "SYSTEM_2", f"High Uncertainty (Entropy {entropy:.2f} > {self.entropy_threshold})"
            
        # 2. Check Stability (Energy Variance)
        # If energy is fluctuating wildly, maybe we are stuck in a loop -> Need Planning
        if len(self.energy_history) >= 5:
            variance = np.var(list(self.energy_history))
            if variance > self.variance_threshold:
                return "SYSTEM_2", f"Unstable Energy (Var {variance:.4f} > {self.variance_threshold})"
        
        # Default: System 1 (Fast, Efficient)
        return "SYSTEM_1", "Stable & Confident"
