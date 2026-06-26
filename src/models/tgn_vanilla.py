import torch
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastNeighborLoader,
    LastAggregator,
)
from torch_geometric.data import TemporalData

from models.base import BaseModel
from utils.negative_sampling import BaseNegativeSampler


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)

        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.lin_final = torch.nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


class TGN(BaseModel):
    def __init__(
        self,
        num_neighbors: int,
        dropout: float,
        memory_dim: int,
        embedding_dim: int,
        time_dim: int,
        data: TemporalData,
        negative_sampler: BaseNegativeSampler,
        labeling: None,
        device: str,
    ):
        super().__init__()
        self._data = data
        self._device = device

        self.neighbor_loader = LastNeighborLoader(
            data.num_nodes, size=num_neighbors, device=self._device
        )

        self.memory = TGNMemory(
            data.num_nodes,
            data.msg.size(-1),
            memory_dim,
            time_dim,
            message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )

        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=data.msg.size(-1),
            time_enc=self.memory.time_enc,
        )

        self.negative_sampler = negative_sampler

        self.link_pred = LinkPredictor(in_channels=embedding_dim)

        self.assoc = torch.empty(data.num_nodes, dtype=torch.long, device=self._device)

    def on_epoch_start(self):
        self.memory.reset_state()
        self.neighbor_loader.reset_state()
        self.negative_sampler.reset()

    def train_step(self, batch, criterion):
        self.negative_sampler.sample(batch=batch, size=batch.src.size(0))
        n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self._device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = self.memory(n_id)
        z = self.gnn(
            z,
            last_update,
            edge_index,
            self._data.t[e_id],
            self._data.msg[e_id],
        )

        pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
        neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        self.memory.update_state(batch.src, batch.dst, batch.t.long(), batch.msg)
        self.neighbor_loader.insert(batch.src, batch.dst)

        return loss

    @torch.no_grad()
    def test_step(self, batch):
        self.negative_sampler.sample(batch=batch, size=batch.src.size(0))
        n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self._device)

        z, last_update = self.memory(n_id)
        z = self.gnn(
            z,
            last_update,
            edge_index,
            self._data.t[e_id],
            self._data.msg[e_id],
        )

        pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
        neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

        self.memory.update_state(batch.src, batch.dst, batch.t.long(), batch.msg)
        self.neighbor_loader.insert(batch.src, batch.dst)

        return pos_out, neg_out

    def after_optimizer_step(self):
        self.memory.detach()
