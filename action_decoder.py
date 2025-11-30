import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActionDecoder(nn.Module):
    """
    The Body.
    Decodes latent thoughts into concrete actions in physical space.
    NO HARDCODED TEMPLATES.
    """
    def __init__(self, latent_dim, output_dim=4, num_actions=6):
        super(ActionDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        
        # Shared trunk
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        # Head 1: Action Logits (What to do)
        # 0: Draw, 1: Symmetrize, 2: Clear, 3: Noise, 4: Random, 5: Inspect
        self.action_head = nn.Linear(64, num_actions)

        # Head 2: Action Parameters (Where/How)
        # [x, y, scale, axis/variant]
        # x, y: [-1, 1]
        # scale: [0, 1]
        # axis: Continuous value to be discretized or used as is
        self.param_head = nn.Linear(64, 4)

        self.action_names = ["DRAW", "SYMMETRIZE", "CLEAR", "NOISE", "RANDOM", "INSPECT"]

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))

        # Head 1
        action_logits = self.action_head(x)
        
        # Head 2
        params = torch.tanh(self.param_head(x)) # Bound to [-1, 1]
        
        return action_logits, params

    def decode_action(self, action_logits, params):
        """
        Interprets the network output into a usable command (Legacy support).
        """
        action_idx = torch.argmax(action_logits).item()
        return self._package_action(action_idx, params)

    def sample_action(self, action_logits, deterministic=False):
        """
        Samples an action from logits.
        """
        if deterministic:
            return torch.argmax(action_logits).item()
        else:
            probs = F.softmax(action_logits, dim=-1)
            return torch.multinomial(probs, 1).item()

    def get_action_name(self, action_id):
        if 0 <= action_id < len(self.action_names):
            return self.action_names[action_id]
        return "UNKNOWN"

    def get_action_code(self, action_id):
        """
        Returns a pseudo-code string for the action.
        """
        name = self.get_action_name(action_id)
        return f"EXECUTE_{name}"

    def _package_action(self, action_idx, params):
        # Unpack parameters
        if isinstance(params, torch.Tensor):
            params = params.detach().cpu().numpy().flatten()

        x, y, p3, p4 = params if len(params) >= 4 else (0,0,0,0)
        
        # Map to action types
        # Note: Original code only had 4 actions. We map indices safely.
        safe_idx = action_idx if action_idx < 4 else 3 # Fallback to NOISE for new actions if not handled
        action_type = ["DRAW", "SYMMETRIZE", "CLEAR", "NOISE"][safe_idx]
        
        return {
            "type": action_type,
            "x": x,
            "y": y,
            "p3": p3, # Scale or Axis
            "p4": p4  # Extra param
        }
