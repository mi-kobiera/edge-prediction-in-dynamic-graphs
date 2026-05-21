import torch

from .base import NodeLabeling
from utils.config import ExperimentConfig

class ZeroNL(NodeLabeling):
    def __init__(self):
        super().__init__()

    def compute(self, z, edge_index, local_src, local_dst):
        """
        Output shape: [Batch_Size, Num_Nodes].
        """
        num_nodes = z.size(0)
        batch_size = local_src.size(0)
        device = z.device

        batch_labels = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)

        return batch_labels
