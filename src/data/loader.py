import os
import logging
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

logger = logging.getLogger(__name__)

def load_tgn_format_dataset(data_dir: str, network_name: str, val_ratio: float = 0.15, test_ratio: float = 0.15) -> TemporalData:
    """
    Wczytuje zbiory danych przetworzone w formacie TGN/TGAT.
    Spodziewa się plików: ml_<network>.csv, ml_<network>.npy, ml_<network_node>.npy
    """
    csv_path = os.path.join(data_dir, f"ml_{network_name}.csv")
    edge_feat_path = os.path.join(data_dir, f"ml_{network_name}.npy")
    node_feat_path = os.path.join(data_dir, f"ml_{network_name}_node.npy")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {csv_path}")

    # 1. Wczytywanie krawędzi z CSV
    df = pd.read_csv(csv_path)
    
    # Przetworzone pliki zazwyczaj mają nagłówki: u, i, ts, label, idx
    if 'u' in df.columns and 'i' in df.columns and 'ts' in df.columns:
        src = torch.tensor(df['u'].values, dtype=torch.long)
        dst = torch.tensor(df['i'].values, dtype=torch.long)
        t = torch.tensor(df['ts'].values, dtype=torch.float32) # float lub long w zależności od modelu
        y = torch.tensor(df['label'].values, dtype=torch.float32) if 'label' in df.columns else torch.zeros(len(df))
    else:
        # Fallback, jeśli nagłówki są inne (zakładamy kolejność: src, dst, ts, label)
        src = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
        dst = torch.tensor(df.iloc[:, 1].values, dtype=torch.long)
        t = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32)
        y = torch.tensor(df.iloc[:, 3].values, dtype=torch.float32)

    # 2. Wczytywanie cech krawędzi (msg)
    if os.path.exists(edge_feat_path):
        edge_features = np.load(edge_feat_path)
        # Zwykle pliki ml_network.npy mają o 1 wiersz więcej (wiersz 0 to zera dla paddingu)
        if 'idx' in df.columns:
            msg = torch.tensor(edge_features[df['idx'].values], dtype=torch.float32)
        else:
            # Jeśli nie ma idx, a plik .npy ma o 1 wiersz więcej niż CSV
            if len(edge_features) == len(df) + 1:
                msg = torch.tensor(edge_features[1:], dtype=torch.float32)
            else:
                msg = torch.tensor(edge_features, dtype=torch.float32)
    else:
        msg = torch.zeros((len(src), 1), dtype=torch.float32) # Dummy features

    # 3. Tworzenie obiektu TemporalData
    data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
    data.e_id = torch.arange(data.num_events)

    # 4. Wczytywanie cech węzłów (jeśli model ich wymaga)
    if os.path.exists(node_feat_path):
        node_features = np.load(node_feat_path)
        data.x = torch.tensor(node_features, dtype=torch.float32)

    # 5. Chronologiczny podział na Train / Val / Test
    num_events = len(src)
    val_time, test_time = np.quantile(t.numpy(), [1 - val_ratio - test_ratio, 1 - test_ratio])
    
    # Tworzenie masek
    data.train_mask = (t <= val_time)
    data.val_mask = (t > val_time) & (t <= test_time)
    data.test_mask = (t > test_time)

    # Upewnijmy się, że typy danych są poprawne i bez problemów wejdą do modelu
    for key in ['src', 'dst', 'e_id']:
        setattr(data, key, getattr(data, key).to(torch.long))

    logger.info(f"Załadowano {network_name}: {num_events} interakcji. "
                f"Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")

    return data


def get_dataloader(
    data: TemporalData, batch_size: int = 1, neg_sampling_ratio: float = 0.0
) -> TemporalDataLoader:
    return TemporalDataLoader(
        data,
        batch_size=batch_size,
        neg_sampling_ratio=neg_sampling_ratio,
        num_workers=0,
    )


# import os
# import logging
# import torch
# from torch_geometric.data import TemporalData
# from torch_geometric.loader import TemporalDataLoader

# logger = logging.getLogger(__name__)


# def load_data(data_path: str) -> TemporalData:
#     if not os.path.exists(data_path):
#         logger.error(f"Data file not found at: '{data_path}'")
#         return

#     data = torch.load(data_path, weights_only=False)
#     data.e_id = torch.arange(data.num_events)
#     return data


# def get_dataloader(
#     data: TemporalData, batch_size: int = 1, neg_sampling_ratio: float = 0.0
# ) -> TemporalDataLoader:
#     return TemporalDataLoader(
#         data,
#         batch_size=batch_size,
#         neg_sampling_ratio=neg_sampling_ratio,
#         num_workers=0,
#     )
