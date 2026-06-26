import torch
from abc import ABC, abstractmethod


class NodeLabeling(ABC):
    @property
    @abstractmethod
    def label_dim(self) -> int:
        pass

    @abstractmethod
    def compute(self, n_id, edge_index, src_idx, dst_idx) -> torch.Tensor:
        raise NotImplementedError
