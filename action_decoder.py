import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionDecoder(nn.Module):
    """
    The Body.
    Decodes latent thoughts into concrete actions in physical space.
    NO HARDCODED TEMPLATES.
    """
    def __init__(self, latent_dim, output_dim=4):
        super(ActionDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Shared trunk
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        # Head 1: Action Logits (What to do)
        # 0: Draw, 1: Symmetrize, 2: Clear, 3: Noise
        self.action_head = nn.Linear(64, 4)

        # Head 2: Action Parameters (Where/How)
        # [x, y, scale, axis/variant]
        # x, y: [-1, 1]
        # scale: [0, 1]
        # axis: Continuous value to be discretized or used as is
        self.param_head = nn.Linear(64, 4)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))

        # Head 1
        action_logits = self.action_head(x)

        # Head 2
        params_raw = self.param_head(x)

        x_coord = torch.tanh(params_raw[:, 0:1])
        y_coord = torch.tanh(params_raw[:, 1:2])
        scale = torch.sigmoid(params_raw[:, 2:3])
        extra = torch.tanh(params_raw[:, 3:4])

        params = torch.cat([x_coord, y_coord, scale, extra], dim=1)
        
        return action_logits, params

    def decode_action(self, action_logits, params):
        """
        Interprets the network output into a usable command.
        """
        action_idx = torch.argmax(action_logits).item()
        
        # Unpack parameters
        x, y, p3, p4 = params.detach().numpy().flatten()
        
        # Map to action types
        action_type = ["DRAW", "SYMMETRIZE", "CLEAR", "NOISE"][action_idx]
        
        return {
            "type": action_type,
            "x": x,
            "y": y,
            "p3": p3, # Scale or Axis
            "p4": p4  # Extra param
        }
