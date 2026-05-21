import torch
from abc import ABC, abstractmethod
from torch_geometric.data import TemporalData
import numpy as np


class BaseNegativeSampler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, batch: TemporalData, size: int, **kwargs):
        pass


class RandomNegariveSampler(BaseNegativeSampler):
    def __init__(self, data: TemporalData):
        super().__init__()
        self.min_src = int(data.src.min())
        self.max_src = int(data.src.max())
        self.min_dst = int(data.dst.min())
        self.max_dst = int(data.dst.max())
        self.device = data.src.device
        self.dtype = data.src.dtype

    def sample(self, batch, size, **kwargs):
        neg_src = torch.randint(low=self.min_src, high=self.max_src + 1, size=(size,), dtype=self.dtype, device=self.device)
        neg_dst = torch.randint(low=self.min_dst, high=self.max_dst + 1, size=(size,), dtype=self.dtype, device=self.device)

        batch.neg_src = neg_src
        batch.neg_dst = neg_dst

        n_ids = [batch.src, batch.dst, neg_src, neg_dst]
        batch.n_id = torch.cat(n_ids, dim=0).unique()


class HistoricalNegativeSampler(BaseNegativeSampler):
    def __init__(self, data: TemporalData):
        super().__init__()
        self.min_dst = int(data.dst.min())
        self.max_dst = int(data.dst.max())
        self.device = data.src.device
        self.dtype = data.src.dtype
        
        self.history = {}

    def reset(self):
        """
        call every single epoch
        """
        self.history = {}

    def sample(self, batch: TemporalData, size: int, **kwargs):
        src_cpu = batch.src.cpu().numpy()
        dst_cpu = batch.dst.cpu().numpy()
        
        neg_dst_list = []
        
        for i in range(size):
            u = src_cpu[i] if i < len(src_cpu) else src_cpu[0]
            
            past_interactions = self.history.get(u, [])
            
            if len(past_interactions) > 0:
                neg_v = np.random.choice(past_interactions)
            else:
                neg_v = np.random.randint(self.min_dst, self.max_dst + 1)
                
            neg_dst_list.append(neg_v)

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
        """
        :param data: Pełny graf (lub graf testowy) do wyciągnięcia min/max id węzłów.
        :param train_data: Dane użyte do treningu - potrzebne, aby zapamiętać, 
                           jakie krawędzie NIE SĄ indukcyjne.
        """
        super().__init__()
        self.min_dst = int(data.dst.min())
        self.max_dst = int(data.dst.max())
        self.device = data.src.device
        self.dtype = data.src.dtype
        
        # Zapisujemy, jakie krawędzie wystąpiły w treningu (u -> set(v))
        self.train_history = self._build_train_history(train_data)
        
        # Historia indukcyjna: przechowuje krawędzie widziane w fazie testowej, 
        # które NIE wystąpiły w train_history
        self.inductive_history = {}

    def _build_train_history(self, train_data: TemporalData):
        train_hist = {}
        src_cpu = train_data.src.cpu().numpy()
        dst_cpu = train_data.dst.cpu().numpy()
        for u, v in zip(src_cpu, dst_cpu):
            if u not in train_hist:
                train_hist[u] = set()
            train_hist[u].add(v)
        return train_hist

    def reset(self):
        """
        Wywołaj przed rozpoczęciem każdej ewaluacji (fazy testowej/walidacyjnej),
        aby wyczyścić zebraną historię z poprzednich epok.
        """
        self.inductive_history = {}

    def sample(self, batch: TemporalData, size: int, **kwargs):
        src_cpu = batch.src.cpu().numpy()
        dst_cpu = batch.dst.cpu().numpy()
        
        neg_dst_list = []
        
        for i in range(size):
            u = src_cpu[i] if i < len(src_cpu) else src_cpu[0]
            
            # Pobieramy dotychczasowe krawędzie indukcyjne dla danego węzła
            past_inductive_interactions = self.inductive_history.get(u, [])
            
            if len(past_inductive_interactions) > 0:
                # Losujemy negatywny węzeł docelowy z krawędzi indukcyjnych (nowych)
                neg_v = np.random.choice(past_inductive_interactions)
            else:
                # Fallback do Random NS, zgodnie z opisem z publikacji:
                # "the remaining negative edges are sampled by the random NS strategy"
                neg_v = np.random.randint(self.min_dst, self.max_dst + 1)
                
            neg_dst_list.append(neg_v)

        neg_dst = torch.tensor(neg_dst_list, dtype=self.dtype, device=self.device)
        # Zachowujemy oryginalne src, zmieniamy tylko dst, symulując nieistniejącą 
        # w danym momencie krawędź
        neg_src = batch.src.clone()[:size]

        batch.neg_src = neg_src
        batch.neg_dst = neg_dst

        n_ids = [batch.src, batch.dst, neg_src, neg_dst]
        batch.n_id = torch.cat(n_ids, dim=0).unique()

        # Aktualizujemy historię indukcyjną o bieżący batch (ważne: po predykcji/samplingu!)
        for u, v in zip(src_cpu, dst_cpu):
            # Sprawdzamy, czy krawędź u->v była widziana podczas treningu
            is_in_train = u in self.train_history and v in self.train_history[u]
            
            if not is_in_train:
                # Jeśli to jest "nowa" krawędź z fazy testowej, zapisujemy ją
                if u not in self.inductive_history:
                    self.inductive_history[u] = []
                self.inductive_history[u].append(v)

