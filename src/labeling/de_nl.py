import torch
from torch_geometric.utils import to_dense_adj
from .base import NodeLabeling

MAX_DIST = 4
# Wartości dystansów: 0..MAX_DIST = osiągalne, MAX_DIST+1 = nieosiągalne (sentinel)
# Embedding potrzebuje MAX_DIST + 2 slotów


class DENL(NodeLabeling):
    def __init__(self):
        super().__init__()

    def compute(self, z, edge_index, local_src, local_dst):
        """
        Distance Encoding (DE) Node Labeling.

        Dla każdego węzła v zwraca parę (d_src, d_dst) gdzie:
            d_src = dist(v, src), d_dst = dist(v, dst)

        Węzły nieosiągalne lub dalsze niż MAX_DIST dostają wartość MAX_DIST + 1.

        Output shape: [Batch_Size, Num_Nodes, 2]
        """
        num_nodes = z.size(0)
        batch_size = local_src.size(0)
        device = z.device

        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)  # [N, N]

        # Sentinel dla nieosiągalnych = MAX_DIST + 1
        sentinel = MAX_DIST + 1
        batch_dists = torch.full(
            (batch_size, num_nodes, 2), sentinel, dtype=torch.long, device=device
        )

        for i in range(batch_size):
            src = local_src[i].item()
            dst = local_dst[i].item()

            dist_src = self._bfs_distances(adj, src, num_nodes, device)  # [N]
            dist_dst = self._bfs_distances(adj, dst, num_nodes, device)  # [N]

            # Clamp: węzły dalej niż MAX_DIST → sentinel
            dist_src = dist_src.clamp(max=sentinel)
            dist_dst = dist_dst.clamp(max=sentinel)

            batch_dists[i, :, 0] = dist_src
            batch_dists[i, :, 1] = dist_dst

        return batch_dists

    @staticmethod
    def _bfs_distances(adj, source, num_nodes, device):
        """
        BFS po gęstej macierzy sąsiedztwa od węzła `source`.
        Nieosiągalne węzły mają dist = num_nodes (+inf sentinel, clampowany później).
        """
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