import torch
from torch.nn import Linear, Embedding
from data import labeling
from models.TGN import TGN


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, max_dist=4):
        super().__init__()
        self.max_dist = max_dist
        
        # Wartości dystansów to: 0, 1, ..., max_dist, oraz max_dist+1 (nieosiągalne).
        # Dlatego rozmiar słownika to max_dist + 2.
        self.label_emb = Embedding(max_dist + 2, in_channels)
        
        self.lin_transform = Linear(in_channels, hidden_channels)

        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)

        self.lin_struct = Linear(hidden_channels, hidden_channels)

        self.lin_final = Linear(hidden_channels, 1)

    def forward(self, z, edge_index, local_src, local_dst):
        # 1. Pobieramy zbatczowane dystanse: [Batch_Size, Num_Nodes, 2]
        # (Zakładam, że zaimportowałeś napisaną wcześniej funkcję batched_de_node_labeling)
        batch_dists = labeling.de_node_labeling(
            z, edge_index, local_src, local_dst, max_dist=self.max_dist
        )

        # 2. Rozdzielamy na dystans do src i dystans do dst
        dist_src = batch_dists[:, :, 0]  # [Batch_Size, Num_Nodes]
        dist_dst = batch_dists[:, :, 1]  # [Batch_Size, Num_Nodes]

        # 3. Tworzymy embeddingi dla dystansów i sumujemy je
        emb_src = self.label_emb(dist_src)  # [Batch_Size, Num_Nodes, Dim]
        emb_dst = self.label_emb(dist_dst)  # [Batch_Size, Num_Nodes, Dim]
        label_feats = emb_src + emb_dst     # [Batch_Size, Num_Nodes, Dim]

        # 4. Łączymy oryginalne cechy węzłów z cechami relacyjnymi (dystansami)
        z_expanded = z.unsqueeze(0).expand(local_src.size(0), -1, -1)
        z_combined = z_expanded + label_feats
        
        # 5. Transformacja węzłów w kontekście badanej krawędzi
        h_nodes = self.lin_transform(z_combined).relu()  # [Batch_Size, Num_Nodes, Hidden]

        # 6. Tworzymy maskę. 
        # Chcemy uwzględnić w strukturze tylko węzły, które są w okolicy max_dist.
        # Warunek (dist <= max_dist) odrzuca węzły "nieosiągalne" (wartość max_dist + 1).
        # Jeśli dodatkowo chcesz ODRZUCIĆ z poolingu same węzły src i dst 
        # (tak jak robiło to batch_labels > 0 w starym kodzie), użyj kodu poniżej:
        
        is_reachable_src = (dist_src > 0) & (dist_src <= self.max_dist)
        is_reachable_dst = (dist_dst > 0) & (dist_dst <= self.max_dist)
        mask = (is_reachable_src | is_reachable_dst).unsqueeze(-1).float() # [Batch_Size, Num_Nodes, 1]

        # 7. Sum pooling ważony maską (zbieramy informację o otoczeniu krawędzi)
        h_struct = (h_nodes * mask).sum(dim=1)  # [Batch_Size, Hidden]

        # 8. Tradycyjne podejście: cechy samych końcówek krawędzi
        z_src = z[local_src]
        z_dst = z[local_dst]

        # 9. Finałowa fuzja (węzeł A + węzeł B + informacja o strukturze między nimi)
        h = self.lin_src(z_src) + self.lin_dst(z_dst) + self.lin_struct(h_struct)
        h = h.relu()

        return self.lin_final(h)


class TGN_DE(TGN):
    def __init__(self, data, cfg):
        super().__init__(data, cfg)
        self.link_pred = LinkPredictor(
            in_channels=cfg.model.embedding_dim, hidden_channels=cfg.model.embedding_dim
        ).to(cfg.device)

    def train_step(self, batch, criterion):
        n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self._device)

        z, last_update = self.memory(n_id)
        z = self.gnn(
            z,
            last_update,
            edge_index,
            self._data.t[e_id].to(self._device),
            self._data.msg[e_id].to(self._device),
        )

        local_src = self.assoc[batch.src]
        local_dst = self.assoc[batch.dst]
        local_neg_dst = self.assoc[batch.neg_dst]

        pos_out = self.link_pred(z, edge_index, local_src, local_dst)
        neg_out = self.link_pred(z, edge_index, local_src, local_neg_dst)

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        self.memory.update_state(batch.src, batch.dst, batch.t.long(), batch.msg)
        self.neighbor_loader.insert(batch.src, batch.dst)

        return loss

    @torch.no_grad()
    def test_step(self, batch):
        n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self._device)

        z, last_update = self.memory(n_id)
        z = self.gnn(
            z,
            last_update,
            edge_index,
            self._data.t[e_id].to(self._device),
            self._data.msg[e_id].to(self._device),
        )

        local_src = self.assoc[batch.src]
        local_dst = self.assoc[batch.dst]
        local_neg_dst = self.assoc[batch.neg_dst]

        pos_out = self.link_pred(z, edge_index, local_src, local_dst)
        neg_out = self.link_pred(z, edge_index, local_src, local_neg_dst)

        self.memory.update_state(batch.src, batch.dst, batch.t.long(), batch.msg)
        self.neighbor_loader.insert(batch.src, batch.dst)

        return pos_out, neg_out
