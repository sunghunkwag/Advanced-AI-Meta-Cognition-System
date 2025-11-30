import numpy as np

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

    def apply_action(self, action_dict):
        """
        Apply the action from the Body to the World.
        """
        act_type = action_dict['type']
        x, y = action_dict['x'], action_dict['y'] # [-1, 1]
        
        # Denormalize coordinates
        c = int((x + 1) / 2 * (self.size - 1))
        r = int((y + 1) / 2 * (self.size - 1))
        
        # Clip
        c = np.clip(c, 0, self.size - 1)
        r = np.clip(r, 0, self.size - 1)

        if act_type == "DRAW":
            self.grid[r, c] = 1.0 # Draw a point
            # Draw a small 3x3 block for visibility
            self.grid[max(0, r-1):min(self.size, r+2), max(0, c-1):min(self.size, c+2)] = 0.8
            
        elif act_type == "CLEAR":
            self.grid.fill(0)
            
        elif act_type == "NOISE":
            noise = np.random.rand(self.size, self.size) * 0.1
            self.grid += noise
            
        elif act_type == "SYMMETRIZE":
            # Simple horizontal symmetry
            self.grid = (self.grid + np.fliplr(self.grid)) / 2

        return self.grid

        return sym_diff

    def calculate_energy(self):
        """
        Calculate the 'Energy' of the world.
        Lower energy = More ordered/symmetrical/clean.
        """
        # 1. Symmetry Energy (Lower is better)
        sym_diff = np.abs(self.grid - np.fliplr(self.grid)).mean()
        
        # 2. Entropy/Boredom Penalty
        # If the world is empty, Energy should be high.
        # Target density: 10% filled.
        density = self.grid.mean()
        target_density = 0.1
        density_penalty = abs(target_density - density) * 5.0
        
        return sym_diff + density_penalty
