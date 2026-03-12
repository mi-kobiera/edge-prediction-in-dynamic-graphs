import torch
from torch_geometric.utils import to_dense_adj


def zot_node_labeling(z, edge_index, local_src, local_dst):
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


def de_node_labeling(z, edge_index, local_src, local_dst, max_dist=3):
    """
    Distance Encoding Labeling
    Output shape: [Batch_Size, Num_Nodes, 2].
    """
    num_nodes = z.size(0)
    batch_size = local_src.size(0)
    device = z.device

    # 1. Zapewnienie inwariantności permutacji (tak jak (dst, src) if src > dst else (src, dst))
    mask_swap = local_src > local_dst
    src = torch.where(mask_swap, local_dst, local_src)
    dst = torch.where(mask_swap, local_src, local_dst)

    # 2. Tworzymy gęstą macierz sąsiedztwa [Num_Nodes, Num_Nodes]
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)

    # 3. Inicjalizacja wyników. Wartość 'max_dist + 1' oznacza "nieosiągalne" w tym promieniu
    dist_src = torch.full(
        (batch_size, num_nodes), max_dist + 1, dtype=torch.long, device=device
    )
    dist_dst = torch.full(
        (batch_size, num_nodes), max_dist + 1, dtype=torch.long, device=device
    )

    # 4. Inicjalizacja wektorów "frontier" (węzły startowe) -> kształt [Batch_Size, Num_Nodes]
    front_src = torch.zeros(batch_size, num_nodes, device=device)
    front_dst = torch.zeros(batch_size, num_nodes, device=device)

    front_src.scatter_(1, src.unsqueeze(1), 1)
    front_dst.scatter_(1, dst.unsqueeze(1), 1)

    # Dystans do samego siebie to 0
    dist_src[front_src.bool()] = 0
    dist_dst[front_dst.bool()] = 0

    # Maski odwiedzonych węzłów (żeby nie nadpisywać krótszych dystansów)
    visited_src = front_src.bool()
    visited_dst = front_dst.bool()

    # 5. Równoległy BFS dla całego batcha przez mnożenie macierzy
    for d in range(1, max_dist + 1):
        # Mnożenie: [Batch_Size, Num_Nodes] @ [Num_Nodes, Num_Nodes] -> [Batch_Size, Num_Nodes]
        front_src = torch.matmul(front_src, adj)
        front_dst = torch.matmul(front_dst, adj)

        # Znajdź nowo odkryte węzły (są w frontierze, ale jeszcze nie odwiedzone)
        new_src = (front_src > 0) & (~visited_src)
        new_dst = (front_dst > 0) & (~visited_dst)

        # Przypisz dystans
        dist_src[new_src] = d
        dist_dst[new_dst] = d

        # Zaktualizuj odwiedzone
        visited_src |= new_src
        visited_dst |= new_dst

        # Przygotuj frontier na kolejną iterację (chcemy propagować tylko z nowo odkrytych)
        front_src = new_src.float()
        front_dst = new_dst.float()

    # 6. Łączymy wyniki. Zwracamy tensor [Batch_Size, Num_Nodes, 2]
    batch_dist = torch.stack([dist_src, dist_dst], dim=-1)

    return batch_dist

def drnl_node_labeling(z, edge_index, local_src, local_dst, max_dist=10):
    """
    Double Radius Node Labeling (DRNL) - Batch Version
    Output shape: [Batch_Size, Num_Nodes].
    """
    num_nodes = z.size(0)
    batch_size = local_src.size(0)
    device = z.device

    # 1. Permutation invariance
    mask_swap = local_src > local_dst
    src = torch.where(mask_swap, local_dst, local_src)
    dst = torch.where(mask_swap, local_src, local_dst)

    # 2. Dense adjacency matrix [Num_Nodes, Num_Nodes]
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)

    # 3. BFS distances from src and dst — shape [Batch_Size, Num_Nodes]
    # Initialize with infinity
    INF = max_dist + 1
    dist_src = torch.full((batch_size, num_nodes), INF, dtype=torch.float, device=device)
    dist_dst = torch.full((batch_size, num_nodes), INF, dtype=torch.float, device=device)

    # Distance to self = 0
    dist_src.scatter_(1, src.unsqueeze(1), 0)
    dist_dst.scatter_(1, dst.unsqueeze(1), 0)

    # 4. Parallel BFS via matrix multiplication
    front_src = torch.zeros(batch_size, num_nodes, device=device)
    front_dst = torch.zeros(batch_size, num_nodes, device=device)
    front_src.scatter_(1, src.unsqueeze(1), 1.0)
    front_dst.scatter_(1, dst.unsqueeze(1), 1.0)

    visited_src = front_src.bool()
    visited_dst = front_dst.bool()

    for d in range(1, max_dist + 1):
        front_src = torch.matmul(front_src, adj)
        front_dst = torch.matmul(front_dst, adj)

        new_src = (front_src > 0) & (~visited_src)
        new_dst = (front_dst > 0) & (~visited_dst)

        dist_src[new_src] = d
        dist_dst[new_dst] = d

        visited_src |= new_src
        visited_dst |= new_dst

        front_src = new_src.float()
        front_dst = new_dst.float()

    # 5. DRNL formula: z = 1 + min(d_src, d_dst) + (d//2) * (d//2 + d%2 - 1)
    #    where d = dist_src + dist_dst
    #    Nodes unreachable from both -> label 0
    dist_sum = dist_src + dist_dst  # [Batch_Size, Num_Nodes]

    dist_over_2 = torch.div(dist_sum, 2, rounding_mode='floor')
    dist_mod_2  = dist_sum % 2

    z_labels = 1.0 + torch.min(dist_src, dist_dst)
    z_labels = z_labels + dist_over_2 * (dist_over_2 + dist_mod_2 - 1)

    # 6. src and dst nodes always get label 1
    z_labels.scatter_(1, src.unsqueeze(1), 1.0)
    z_labels.scatter_(1, dst.unsqueeze(1), 1.0)

    # 7. Unreachable nodes (INF distances) -> label 0
    unreachable = (dist_src >= INF) | (dist_dst >= INF)
    z_labels[unreachable] = 0.0

    return z_labels.to(torch.long)
