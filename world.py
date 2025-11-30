import numpy as np

class World:
    """
    The Environment.
    A simple 2D grid where the agent draws.
    """
    def __init__(self, size=32):
        self.size = size
        self.grid = np.zeros((size, size))
        self.global_context = {'grid': self.grid}

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

        self.global_context['grid'] = self.grid
        return self.grid

    def calculate_energy(self):
        """
        Calculate the 'Energy' of the world.
        Lower energy = More ordered/symmetrical/clean.
        """
        # 1. Symmetry Energy (Lower is better)
        sym_diff = np.abs(self.grid - np.fliplr(self.grid)).mean()
        
        # 2. Entropy (Sparseness is preferred?)
        # Let's say we want a specific pattern, but for now, symmetry is the goal.
        
        return sym_diff

class InternalSandbox(World):
    def __init__(self, size=32):
        super().__init__(size)
        self.global_context = {'grid': self.grid}

    def execute(self, code):
        """
        Executes the code string from the Action Decoder.
        """
        try:
            # Map pseudo-code to actual actions
            if "DRAW" in code:
                # Need params. Random for now or extract from context if passed?
                # The code string is just "EXECUTE_DRAW" from ActionDecoder.
                # We need to access the params somehow.
                # But ActionDecoder separates type and params.
                # main_asi passes code only.
                # Let's assume random params for "Draw" if not specified,
                # or simplified behavior (draw center).
                action = {"type": "DRAW", "x": 0, "y": 0}
                self.apply_action(action)
                return "Draw executed at center", True

            elif "SYMMETRIZE" in code:
                action = {"type": "SYMMETRIZE", "x": 0, "y": 0}
                self.apply_action(action)
                return "Symmetrize executed", True

            elif "CLEAR" in code:
                action = {"type": "CLEAR", "x": 0, "y": 0}
                self.apply_action(action)
                return "Clear executed", True

            elif "NOISE" in code:
                action = {"type": "NOISE", "x": 0, "y": 0}
                self.apply_action(action)
                return "Noise injected", True

            elif "RANDOM" in code:
                 # Pick random action
                 acts = ["DRAW", "SYMMETRIZE", "CLEAR", "NOISE"]
                 import random
                 t = random.choice(acts)
                 action = {"type": t, "x": random.uniform(-1,1), "y": random.uniform(-1,1)}
                 self.apply_action(action)
                 return f"Random action {t} executed", True

            elif "INSPECT" in code:
                 return "Inspection complete (No change)", True

            else:
                return f"Unknown command: {code}", False

        except Exception as e:
            return f"Error: {e}", False
