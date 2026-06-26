import torch
from abc import ABC, abstractmethod
from torch_geometric.data import TemporalData
import numpy as np
import copy


class BaseNegativeSampler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, batch: TemporalData, size: int, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass


class RandomNegativeSampler(BaseNegativeSampler):
    def __init__(self, data: TemporalData):
        super().__init__()
        self.min_src = int(data.src.min())
        self.max_src = int(data.src.max())
        self.min_dst = int(data.dst.min())
        self.max_dst = int(data.dst.max())
        self.device = data.src.device
        self.dtype = data.src.dtype

    def sample(self, batch, size, **kwargs):
        device = batch.src.device
        neg_src = torch.randint(
            low=self.min_src,
            high=self.max_src + 1,
            size=(size,),
            dtype=self.dtype,
            device=device,
        )
        neg_dst = torch.randint(
            low=self.min_dst,
            high=self.max_dst + 1,
            size=(size,),
            dtype=self.dtype,
            device=device,
        )

        batch.neg_src = neg_src
        batch.neg_dst = neg_dst

        n_ids = [batch.src, batch.dst, neg_src, neg_dst]
        batch.n_id = torch.cat(n_ids, dim=0).unique()

    def reset(self):
        pass


class HistoricalNegativeSampler(BaseNegativeSampler):
    def __init__(self, data: TemporalData, train_data: TemporalData = None):
        """
        if training sampler set train_data to None
        """
        super().__init__()
        self.min_dst = int(data.dst.min())
        self.max_dst = int(data.dst.max())
        self.device = data.src.device
        self.dtype = data.src.dtype

        self.base_history = {}
        if train_data is not None:
            src_train = train_data.src.cpu().numpy()
            dst_train = train_data.dst.cpu().numpy()

            for u, v in zip(src_train, dst_train):
                if u not in self.base_history:
                    self.base_history[u] = set()
                self.base_history[u].add(v)

            for u in self.base_history:
                self.base_history[u] = np.array(
                    list(self.base_history[u]), dtype=np.int64
                )

        self.history = {}
        self.reset()

    def reset(self):
        """
        call each val epoch
        """
        self.history = copy.deepcopy(self.base_history)

        for u in self.history:
            self.history[u] = self.history[u].tolist()

    def sample(self, batch: TemporalData, size: int, **kwargs):
        src_cpu = batch.src.cpu().numpy()
        dst_cpu = batch.dst.cpu().numpy()

        neg_dst_list = np.empty(size, dtype=np.int64)
        rand_fallback = np.random.randint(self.min_dst, self.max_dst + 1, size=size)

        for i in range(size):
            u = src_cpu[i] if i < len(src_cpu) else src_cpu[0]
            true_v = dst_cpu[i]

            past = self.history.get(u)

            if past:
                valid_past = np.unique([v for v in past if v != true_v])

                if len(valid_past) > 0:
                    idx = np.random.randint(0, len(valid_past))
                    neg_dst_list[i] = valid_past[idx]
                else:
                    neg_dst_list[i] = rand_fallback[i]
            else:
                neg_dst_list[i] = rand_fallback[i]

        neg_dst = torch.tensor(neg_dst_list, dtype=self.dtype, device=self.device)
        neg_src = batch.src.clone()[:size]

        batch.neg_src = neg_src
        batch.neg_dst = neg_dst

        n_ids = [batch.src, batch.dst, neg_src, neg_dst]
        batch.n_id = torch.cat(n_ids, dim=0).unique()

        for u, v in zip(src_cpu, dst_cpu):
            if u not in self.history:
                self.history[u] = []
            self.history[u].append(v)


class InductiveNegativeSampler(BaseNegativeSampler):
    def __init__(self, data: TemporalData, train_data: TemporalData):
        super().__init__()
        self.min_dst = int(data.dst.min())
        self.max_dst = int(data.dst.max())
        self.device = data.src.device
        self.dtype = data.src.dtype

        src_train = train_data.src.cpu().numpy()
        dst_train = train_data.dst.cpu().numpy()
        self.train_edges = set(zip(src_train, dst_train))

        self.inductive_history = {}

    def reset(self):
        """
        call each val epoch
        """
        self.inductive_history = {}

    def sample(self, batch: TemporalData, size: int, **kwargs):
        src_cpu = batch.src.cpu().numpy()
        dst_cpu = batch.dst.cpu().numpy()

        neg_dst_list = np.empty(size, dtype=np.int64)
        rand_floats = np.random.rand(size)
        rand_fallback = np.random.randint(self.min_dst, self.max_dst + 1, size=size)

        for i in range(size):
            u = src_cpu[i] if i < len(src_cpu) else src_cpu[0]

            past = self.inductive_history.get(u)

            if past:
                idx = int(rand_floats[i] * len(past))
                neg_dst_list[i] = past[idx]
            else:
                neg_dst_list[i] = rand_fallback[i]

        neg_dst = torch.tensor(neg_dst_list, dtype=self.dtype, device=self.device)
        neg_src = batch.src.clone()[:size]

        batch.neg_src = neg_src
        batch.neg_dst = neg_dst

        n_ids = [batch.src, batch.dst, neg_src, neg_dst]
        batch.n_id = torch.cat(n_ids, dim=0).unique()

        for u, v in zip(src_cpu, dst_cpu):
            if (u, v) not in self.train_edges:
                if u not in self.inductive_history:
                    self.inductive_history[u] = []
                self.inductive_history[u].append(v)
