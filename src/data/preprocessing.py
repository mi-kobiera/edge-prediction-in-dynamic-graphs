import logging
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import TemporalData

logger = logging.getLogger(__name__)


def preprocess(data_name) -> None:
    """return list of events"""
    logger.info(f"Preprocessing dataset: {data_name}")

    PATH = f"./data/raw/{data_name}.csv"
    OUT_DATA = f"./data/processed/ml_{data_name}.pt"
    OUT_NODE = f"./data/processed/ml_{data_name}_node.pt"

    os.makedirs("./data/processed", exist_ok=True)

    # --- parse ---
    src_l, dst_l, t_l, y_l, feat_l = [], [], [], [], []

    with open(PATH) as f:
        next(f)  # skip header

        for line in f:
            line = line.strip().split(",")

            src_l.append(int(line[0]))
            dst_l.append(int(line[1]))
            t_l.append(float(line[2]))
            y_l.append(int(line[3]))
            feat_l.append(np.array([float(x) for x in line[4:]]))

    df = pd.DataFrame({"src": src_l, "dst": dst_l, "t": t_l, "y": y_l, "feat": feat_l})

    # --- reindex nodes ---
    all_nodes = pd.concat([df.src, df.dst]).unique()
    node2id = {n: i + 1 for i, n in enumerate(all_nodes)}  # 1-indexed

    df["src"] = df["src"].map(node2id)
    df["dst"] = df["dst"].map(node2id)

    # --- sort by time ---
    df = df.sort_values("t").reset_index(drop=True)

    # --- create dataset ---
    data = TemporalData(
        src=torch.tensor(df.src.values, dtype=torch.long),
        dst=torch.tensor(df.dst.values, dtype=torch.long),
        t=torch.tensor(df.t.values, dtype=torch.float),
        msg=torch.tensor(np.stack(df.feat.values), dtype=torch.float),
        y=torch.tensor(df.y.values, dtype=torch.long),
    )

    # --- save data ---
    torch.save(data, OUT_DATA)
    torch.save(node2id, OUT_NODE)

    logger.info(f"Saved TemporalData -> {OUT_DATA}")
    logger.info(f"{data.num_events} events")
    logger.info(f"{data.num_edges} edges")
    logger.info(f"{data.num_nodes} nodes")
    logger.info(f"Saved node mapping -> {OUT_NODE}")


if __name__ == "__main__":
    preprocess("wikipedia")
