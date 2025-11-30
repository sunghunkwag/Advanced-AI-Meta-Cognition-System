from vision import VisionSystem

# Mock GNNObjectExtractor to mimic expected behavior for main_asi.py
class GNNObjectExtractor(VisionSystem):
    def __init__(self, max_objects=5, feature_dim=4):
        super().__init__()
        self.max_objects = max_objects
        self.feature_dim = feature_dim

    def __call__(self, grid):
        # Adapt perceive to match what main_asi expects
        # main_asi expects grid as tensor input
        if grid.dim() == 3: # (C, H, W) or (1, H, W)
            grid = grid.squeeze(0)

        # Use existing perceive
        nodes, adj = self.perceive(grid.numpy())

        # Pad or truncate to max_objects
        num_nodes = nodes.shape[0]
        if num_nodes > self.max_objects:
            nodes = nodes[:self.max_objects]
            adj = adj[:self.max_objects, :self.max_objects]
        elif num_nodes < self.max_objects:
            pad_n = self.max_objects - num_nodes
            # Pad nodes
            nodes_pad = torch.zeros(pad_n, nodes.shape[1])
            nodes = torch.cat([nodes, nodes_pad], dim=0)
            # Pad adj
            adj_pad = torch.zeros(self.max_objects, self.max_objects)
            adj_pad[:num_nodes, :num_nodes] = adj
            adj = adj_pad

        # Ensure feature dim matches (currently 3: y, x, val)
        # main_asi expects 4. Let's pad feature dim.
        if nodes.shape[1] < self.feature_dim:
            nodes = torch.cat([nodes, torch.zeros(self.max_objects, self.feature_dim - nodes.shape[1])], dim=1)

        # main_asi expects batched output (1, N, F)
        return nodes.unsqueeze(0), adj.unsqueeze(0)
