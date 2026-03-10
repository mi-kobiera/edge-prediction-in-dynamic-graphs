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
