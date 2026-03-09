import torch
from torch.nn import Linear, Embedding
from data import labeling
from models.TGN import TGN


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.label_emb = Embedding(3, in_channels)
        self.lin_transform = Linear(in_channels, hidden_channels)

        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)

        self.lin_struct = Linear(hidden_channels, hidden_channels)

        self.lin_final = Linear(hidden_channels, 1)

    def forward(self, z, edge_index, local_src, local_dst):
        batch_labels = labeling.zero_one_two(
            z, edge_index, local_src, local_dst
        )  # [Batch, N]
        label_feats = self.label_emb(batch_labels)  # [Batch, N, Dim]

        z_expanded = z.unsqueeze(0).expand(local_src.size(0), -1, -1)  # [Batch, N, Dim]

        z_combined = z_expanded + label_feats
        h_nodes = self.lin_transform(z_combined).relu()  # [Batch, N, Hidden]

        mask = (batch_labels > 0).unsqueeze(-1).float()  # [Batch, N, 1]

        # Sum pooling ważony maską (tylko sąsiedzi mają wpływ na strukturę)
        h_struct = (h_nodes * mask).sum(dim=1)  # [Batch, Hidden]

        z_src = z[local_src]
        z_dst = z[local_dst]

        h = self.lin_src(z_src) + self.lin_dst(z_dst) + self.lin_struct(h_struct)
        h = h.relu()

        return self.lin_final(h)


class TGN_ZeroOneTwoLT(TGN):
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
