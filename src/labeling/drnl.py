import torch
from torch_geometric.utils import to_dense_adj
from .base import NodeLabeling

MAX_DIST = 10
MAX_LABEL = 101


class DRNL(NodeLabeling):
    def __init__(self):
        super().__init__()

    @property
    def label_dim(self) -> int:
        return 102

    def compute(self, z, edge_index, local_src, local_dst):
        """
        Double Radius Node Labeling (DRNL)

        label = 1 + min(d_src, d_dst) + (d_sum//2) * (d_sum//2 + d_sum%2 - 1)

        Output shape: [Batch_Size, Num_Nodes]
        """
        num_nodes = z.size(0)
        batch_size = local_src.size(0)
        device = z.device

        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)  # [N, N]

        batch_labels = torch.zeros(
            batch_size, num_nodes, dtype=torch.long, device=device
        )

        for i in range(batch_size):
            src = local_src[i].item()
            dst = local_dst[i].item()

            dist_src = self._bfs_distances(adj, src, num_nodes, device)  # [N]
            dist_dst = self._bfs_distances(adj, dst, num_nodes, device)  # [N]

            reachable = (dist_src <= MAX_DIST) & (dist_dst <= MAX_DIST)

            d_src = dist_src[reachable]
            d_dst = dist_dst[reachable]
            d_sum = d_src + d_dst

            over2 = d_sum // 2
            labels = 1 + torch.min(d_src, d_dst) + over2 * (over2 + d_sum % 2 - 1)

            labels = labels.clamp(max=MAX_LABEL)

            node_labels = torch.zeros(num_nodes, dtype=torch.long, device=device)
            node_labels[reachable] = labels

            node_labels[src] = 0
            node_labels[dst] = 0

            batch_labels[i] = node_labels

        return batch_labels

    @staticmethod
    def _bfs_distances(adj, source, num_nodes, device):
        dist = torch.full((num_nodes,), num_nodes, dtype=torch.long, device=device)
        dist[source] = 0

        frontier = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        frontier[source] = True

        for d in range(1, MAX_DIST + 1):
            next_frontier = (adj[frontier].sum(dim=0) > 0) & (dist == num_nodes)
            if not next_frontier.any():
                break
            dist[next_frontier] = d
            frontier = next_frontier

        return dist
