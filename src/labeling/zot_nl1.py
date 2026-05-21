import torch

from .base import NodeLabeling
from utils.config import ExperimentConfig

class ZeroOneTwoNL(NodeLabeling):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.label_emb = torch.nn.Embedding(3, cfg.labeling.label_dim)

    def compute(self, n_id, edge_index, src_idx, dst_idx):
        """
        n_id:       [N]
        edge_index: [2, E]
        src_idx:    scalar
        dst_idx:    scalar

        Returns:    [N, label_dim]
        """

        num_nodes = n_id.size(0)
        device = n_id.device

        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        adj[edge_index[0], edge_index[1]] = True
        adj[edge_index[1], edge_index[0]] = True # undirected

        neigh_src = adj[src_idx] # [N]
        neigh_dst = adj[dst_idx] # [N]

        common_mask = neigh_src & neigh_dst
        other_neigh_mask = (neigh_src | neigh_dst) & ~common_mask

        labels = torch.zeros(num_nodes, dtype=torch.long, device=device)
        labels[other_neigh_mask] = 2
        labels[common_mask] = 1

        labels[src_idx] = 0
        labels[dst_idx] = 0

        return self.label_emb(labels)
