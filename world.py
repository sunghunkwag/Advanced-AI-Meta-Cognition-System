import numpy as np
import torch

class World:
    """
    The Environment.
    A simple 2D grid where the agent draws.
    """
    def __init__(self, size=32):
        self.size = size
        self.grid = np.zeros((size, size))

    def get_state(self):
        return self.grid

    def get_state_tensor(self) -> torch.Tensor:
        """Return the current grid as a torch tensor for models."""
        return torch.tensor(self.grid, dtype=torch.float32)

    def set_state(self, state: torch.Tensor):
        """Set grid from a tensor state (used for simulated rollouts)."""
        array = state.detach().cpu().numpy()
        self.grid = np.clip(array, 0, 1)
        return self.grid

    def apply_action(self, action_dict):
        """
        Apply the action from the Body to the World.
        """
        # Input validation
        if not isinstance(action_dict, dict):
            print(f"[WORLD ERROR] Invalid action type: {type(action_dict)}")
            return self.grid

        if 'type' not in action_dict:
            print(f"[WORLD ERROR] Missing 'type' in action: {action_dict}")
            return self.grid

        act_type = action_dict['type']

        valid_actions = ["DRAW", "SYMMETRIZE", "CLEAR", "NOISE"]
        if act_type not in valid_actions:
            print(f"[WORLD ERROR] Unknown action: {act_type}")
            return self.grid

        # Denormalize coordinates if present
        x = action_dict.get('x', 0)
        y = action_dict.get('y', 0)

        c = int((x + 1) / 2 * (self.size - 1))
        r = int((y + 1) / 2 * (self.size - 1))
        
        # Clip
        c = np.clip(c, 0, self.size - 1)
        r = np.clip(r, 0, self.size - 1)

        if act_type == "DRAW":
            # Draw single pixel with slight blur
            self.grid[r, c] = 1.0

            if r + 1 < self.size:
                self.grid[r+1, c] = 0.5
            if c + 1 < self.size:
                self.grid[r, c+1] = 0.5
            if r + 1 < self.size and c + 1 < self.size:
                self.grid[r+1, c+1] = 0.3
            
        elif act_type == "CLEAR":
            self.grid.fill(0)
            
        elif act_type == "NOISE":
            noise = np.random.rand(self.size, self.size) * 0.1
            self.grid += noise
            self.grid = np.clip(self.grid, 0, 1)  # Prevent overflow
            
        elif act_type == "SYMMETRIZE":
            # Simple horizontal symmetry
            self.grid = (self.grid + np.fliplr(self.grid)) / 2

        # Ensure all grid values stay in valid range [0, 1]
        self.grid = np.clip(self.grid, 0, 1)
        return self.grid


    def calculate_energy(self):
        """
        Calculate the 'Energy' of the world.
        Lower energy = More ordered/symmetrical/clean.
        """
        # 1. Symmetry Energy (Lower is better)
        sym_diff = np.abs(self.grid - np.fliplr(self.grid)).mean()
        
        # 2. Density-based Penalty
        # Count occupied cells (threshold > 0.1) vs total cells
        occupied_ratio = np.sum(self.grid > 0.1) / (self.size * self.size)
        target_density = 0.15
        density_error = abs(target_density - occupied_ratio)

        # More balanced penalty function
        if density_error > 0.1:
            density_penalty = density_error * 3.0 + (density_error ** 2) * 10.0
        else:
            density_penalty = density_error * 2.0

        return (sym_diff * 0.7) + (density_penalty * 0.3)
