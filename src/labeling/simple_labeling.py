import torch

from .base import NodeLabeling
from utils.config import ExperimentConfig

class SimpleLabeling(NodeLabeling):
    def __init__(self, cfg: ExperimentConfig):
        self.label_emb = torch.nn.Embedding(3, cfg.labeling.label_dim)

    def compute(self, batch, n_id, edge_index, assoc):
        z_labels = torch.zeros(n_id.size(0), dtype=torch.long, device=batch.src.device)
        z_labels[assoc[batch.src]] = 1
        z_labels[assoc[batch.dst]] = 2
        z_labels[assoc[batch.neg_dst]] = 2
        return self.label_emb(z_labels)
