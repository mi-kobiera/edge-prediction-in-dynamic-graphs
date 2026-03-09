import torch
from torch_geometric.utils import to_dense_adj


def zero_one_two(z, edge_index, local_src, local_dst):
    """
    Output shape [Batch_Size, Num_Nodes].
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
