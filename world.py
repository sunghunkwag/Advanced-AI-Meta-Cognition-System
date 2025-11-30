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
        x, y = action_dict['x'], action_dict['y']  # [-1, 1]
        
        # Denormalize coordinates
        c = int((x + 1) / 2 * (self.size - 1))
        r = int((y + 1) / 2 * (self.size - 1))
        
        # Clip
        c = np.clip(c, 0, self.size - 1)
        r = np.clip(r, 0, self.size - 1)

        if act_type == "DRAW":
            # Reduced intensity for better control
            self.grid[r, c] = min(1.0, self.grid[r, c] + 0.5)
            # Draw a small 2x2 block for visibility
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        self.grid[nr, nc] = min(1.0, self.grid[nr, nc] + 0.3)
            
        elif act_type == "CLEAR":
            # Clear specific region instead of entire grid
            radius = 2
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        self.grid[nr, nc] = max(0, self.grid[nr, nc] - 0.4)
            
        elif act_type == "NOISE":
            noise = np.random.rand(self.size, self.size) * 0.1
            self.grid = np.clip(self.grid + noise, 0, 1)
            
        elif act_type == "SYMMETRIZE":
            # Simple horizontal symmetry
            self.grid = (self.grid + np.fliplr(self.grid)) / 2

        return self.grid

    def calculate_energy(self):
        """
        Calculate the 'Energy' of the world.
        Lower energy = More ordered/symmetrical/clean.
        
        NOTE: This returns a Python float (not differentiable).
        For gradient-based learning, use REINFORCE algorithm.
        """
        # 1. Symmetry Energy (Lower is better)
        sym_diff = np.abs(self.grid - np.fliplr(self.grid)).sum()
        
        # 2. Density Penalty
        # Target density: 10% filled.
        density = self.grid.sum() / (self.size ** 2)
        target_density = 0.1
        density_penalty = abs(target_density - density) * 20.0
        
        # 3. Over-saturation penalty
        if density > 0.5:
            density_penalty += (density - 0.5) ** 2 * 50
        
        return float(sym_diff + density_penalty)
