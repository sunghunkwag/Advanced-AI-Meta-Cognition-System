import torch
import numpy as np

class VisionSystem:
    """
    The Perception Layer.
    Extracts objects (Nodes) and spatial relationships (Edges) from the raw world grid.
    """
    def __init__(self):
        self.input_dim = 1  # Grayscale/Single channel input

    def perceive(self, world_state):
        """
        Convert raw grid to a graph representation.
        Args:
            world_state (np.array): The grid environment.
        Returns:
            nodes (torch.Tensor): Node features (N, F).
            adj_matrix (torch.Tensor): Adjacency matrix (N, N).
        """
        # 1. Object Extraction (Simplified: Non-zero pixels are objects)
        # In a real system, this would be a CNN.
        objects = np.argwhere(world_state > 0)
        
        if len(objects) == 0:
            # Return dummy graph if empty
            return torch.zeros((1, 3)), torch.zeros((1, 1))

        # Normalize coordinates to [-1, 1]
        h, w = world_state.shape
        nodes = []
        for r, c in objects:
            val = world_state[r, c]
            # Feature: [y, x, value]
            norm_y = (r / h) * 2 - 1
            norm_x = (c / w) * 2 - 1
            nodes.append([norm_y, norm_x, val])
        
        nodes = torch.tensor(nodes, dtype=torch.float32)

        # 2. Relationship Extraction (Edges based on distance)
        # Fully connected for small number of objects, or k-NN
        num_nodes = len(nodes)
        adj_matrix = torch.eye(num_nodes) # Self-loops
        
        # Simple distance-based edges
        if num_nodes > 1:
            coords = nodes[:, :2]
            dists = torch.cdist(coords, coords)
            # Connect if close enough (e.g., within 0.5 normalized distance)
            adj_matrix = (dists < 0.5).float()

        return nodes, adj_matrix

# Mock GNNObjectExtractor to mimic expected behavior for main_asi.py
class GNNObjectExtractor(VisionSystem):
    def __init__(self, max_objects=5, feature_dim=4):
        super().__init__()
        self.max_objects = max_objects
        self.feature_dim = feature_dim

    def __call__(self, grid):
        # Adapt perceive to match what main_asi expects
        # main_asi expects grid as tensor input
        if isinstance(grid, torch.Tensor):
            grid = grid.detach().cpu().numpy()

        if grid.ndim == 3: # (C, H, W) or (1, H, W)
            grid = grid[0]

        # Use existing perceive
        nodes, adj = self.perceive(grid)

        # Pad or truncate to max_objects
        num_nodes = nodes.shape[0]
        if num_nodes > self.max_objects:
            nodes = nodes[:self.max_objects]
            adj = adj[:self.max_objects, :self.max_objects]
        elif num_nodes < self.max_objects:
            pad_n = self.max_objects - num_nodes
            # Pad nodes
            if pad_n > 0:
                nodes_pad = torch.zeros(pad_n, nodes.shape[1])
                nodes = torch.cat([nodes, nodes_pad], dim=0)
                # Pad adj
                adj_pad = torch.zeros(self.max_objects, self.max_objects)
                adj_pad[:num_nodes, :num_nodes] = adj
                adj = adj_pad
            else:
                 # Should not happen given < check
                 pass

        # Ensure feature dim matches (currently 3: y, x, val)
        # main_asi expects 4. Let's pad feature dim.
        if nodes.shape[1] < self.feature_dim:
            nodes = torch.cat([nodes, torch.zeros(self.max_objects, self.feature_dim - nodes.shape[1])], dim=1)

        # main_asi expects batched output (1, N, F)
        return nodes.unsqueeze(0), adj.unsqueeze(0)
