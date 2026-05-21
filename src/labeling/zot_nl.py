import torch
from torch_geometric.utils import to_dense_adj

from .base import NodeLabeling
from utils.config import ExperimentConfig

class ZeroOneTwoNL(NodeLabeling):
    def __init__(self):
        super().__init__()

    def compute(self, z, edge_index, local_src, local_dst):
        """
        Zero-One-Two Labeling 
        Output shape: [Batch_Size, Num_Nodes].
        """
        num_nodes = z.size(0)
        batch_size = local_src.size(0)
        device = z.device

        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)

        neigh_src = adj[local_src]
        neigh_dst = adj[local_dst]

        common = neigh_src * neigh_dst
        union = (neigh_src + neigh_dst) > 0

        batch_labels = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)

        batch_labels[union] = 1
        batch_labels[common.bool()] = 2

        mask_src = torch.zeros_like(batch_labels).scatter_(1, local_src.unsqueeze(1), 1)
        mask_dst = torch.zeros_like(batch_labels).scatter_(1, local_dst.unsqueeze(1), 1)

        batch_labels[mask_src.bool()] = 0
        batch_labels[mask_dst.bool()] = 0

        return batch_labels

    # def compute(self, num_nodes, edge_index, src_idx, dst_idx):
    #     """
    #     num_nodes:
    #     edge_index: [2, E]
    #     src_idx:    [B]
    #     dst_idx:    [B]

    #     Returns:    [B, num_nodes]
    #     """
    #     device = edge_index.device
    #     batch_size = src_idx.size(0)

    #     adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    #     adj[edge_index[0], edge_index[1]] = True
    #     adj[edge_index[1], edge_index[0]] = True # undirected

    #     neigh_src = adj[src_idx] 
    #     neigh_dst = adj[dst_idx] 

    #     common_mask = neigh_src & neigh_dst
    #     other_neigh_mask = (neigh_src | neigh_dst) & ~common_mask

    #     labels = torch.zeros((batch_size, num_nodes), dtype=torch.long, device=device)
    #     labels[other_neigh_mask] = 2
    #     labels[common_mask] = 1

    #     batch_indices = torch.arange(batch_size, device=device)
    #     labels[batch_indices, src_idx] = 0
    #     labels[batch_indices, dst_idx] = 0

    #     return labels