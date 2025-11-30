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
