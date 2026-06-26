import torch
from torch_geometric.utils import to_dense_adj
from .base import NodeLabeling

MAX_DIST = 4


class DE(NodeLabeling):
    def __init__(self):
        super().__init__()

    @property
    def label_dim(self) -> int:
        return 6

    def compute(self, z, edge_index, local_src, local_dst):
        """
        Distance Encoding (DE).

        For each node v returns pair (d_src, d_dst), where:
            d_src = dist(v, src), d_dst = dist(v, dst)

        Nodes unreachable gets MAX_DIST + 1.

        Output shape: [Batch_Size, Num_Nodes, 2]
        """
        num_nodes = z.size(0)
        batch_size = local_src.size(0)
        device = z.device

        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)  # [N, N]

        sentinel = MAX_DIST + 1
        batch_dists = torch.full(
            (batch_size, num_nodes, 2), sentinel, dtype=torch.long, device=device
        )

        for i in range(batch_size):
            src = local_src[i].item()
            dst = local_dst[i].item()

            dist_src = self._bfs_distances(adj, src, num_nodes, device)  # [N]
            dist_dst = self._bfs_distances(adj, dst, num_nodes, device)  # [N]

            dist_src = dist_src.clamp(max=sentinel)
            dist_dst = dist_dst.clamp(max=sentinel)

            batch_dists[i, :, 0] = dist_src
            batch_dists[i, :, 1] = dist_dst

        return batch_dists

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
