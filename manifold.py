import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class GraphAttentionManifold(nn.Module):
    """
    The Mind.
    Uses Graph Attention Networks (GAT) to reason about relationships.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.3, alpha=0.2, nheads=2, truth_vector=None, reject_vector=None):
        super(GraphAttentionManifold, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        
        # Axiom Injection: Truth Vector
        # This represents the "ideal" logical state or consistency check
        if truth_vector is not None:
            self.register_buffer('truth_vector', truth_vector)
        else:
            self.register_buffer('truth_vector', torch.randn(nclass))

        if reject_vector is not None:
            self.register_buffer('reject_vector', reject_vector)
        else:
            self.register_buffer('reject_vector', torch.randn(nclass))

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        
        # Latent State Representation (z)
        z = torch.mean(x, dim=0, keepdim=True) # Global pooling -> (1, D)
        return z

    def check_consistency(self, z):
        """
        Compare current thought (z) with Axiom (Truth).
        Returns a consistency score (0 to 1).
        """
        # z: (1, D)
        # truth_vector: (D) -> (1, D)
        target = self.truth_vector.unsqueeze(0)
        similarity = F.cosine_similarity(z, target, dim=1)
        return (similarity + 1) / 2  # Normalize to [0, 1]

    def check_rejection(self, z):
        """
        Measure distance from rejection vector (contradictions, fallacies).
        Returns penalty score (0 to 1), higher = closer to bad states.
        """
        target = self.reject_vector.unsqueeze(0)
        similarity = F.cosine_similarity(z, target, dim=1)
        return (similarity + 1) / 2
