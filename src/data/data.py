import os
import logging
import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

logger = logging.getLogger(__name__)


def load_data(data_path: str) -> TemporalData:
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at: '{data_path}'")
        return

    data = torch.load(data_path, weights_only=False)
    data.e_id = torch.arange(data.num_events)
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
