import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


def build_temporal_edgelist(dataset, interval_size=86400):
    """
    Konwersja tgbl-review -> format używany przez TEA.

    interval_size:
        86400  = 1 dzień
        604800 = 1 tydzień
        2592000 = ~1 miesiąc
    """

    data = dataset.get_TemporalData()

    src = data.src.numpy()
    dst = data.dst.numpy()
    ts = data.t.numpy()

    temporal_edgelist = {}

    for u, v, t in zip(src, dst, ts):

        ts_bin = int(t // interval_size)

        if ts_bin not in temporal_edgelist:
            temporal_edgelist[ts_bin] = {}

        edge = (str(int(u)), str(int(v)))

        if edge not in temporal_edgelist[ts_bin]:
            temporal_edgelist[ts_bin][edge] = 1
        else:
            temporal_edgelist[ts_bin][edge] += 1

    return temporal_edgelist


def process_edgelist_per_timestamp(temp_edgelist):

    unique_ts = sorted(temp_edgelist.keys())

    node_dict = {}

    for _, e_dict in temp_edgelist.items():
        for e in e_dict:
            node_dict[e[0]] = 1
            node_dict[e[1]] = 1

    num_nodes = len(node_dict)
    num_e_fully_connected = num_nodes * (num_nodes - 1)

    edge_frequency_dict = {}
    ts_edges_dist = []

    seen_edges = set()

    for curr_t in unique_ts:

        curr_edges = set(temp_edgelist[curr_t].keys())

        repeated = len(curr_edges & seen_edges)
        new = len(curr_edges - seen_edges)
        not_repeated = len(seen_edges - curr_edges)

        for e in curr_edges:
            edge_frequency_dict[e] = edge_frequency_dict.get(e, 0) + 1

        ts_edges_dist.append({
            "ts": curr_t,
            "new": new,
            "repeated": repeated,
            "not_repeated": not_repeated,
            "total_curr_ts": len(curr_edges),
            "total_seen_until_curr_ts": len(seen_edges | curr_edges)
        })

        seen_edges |= curr_edges

    return ts_edges_dist, edge_frequency_dict


def plot_tea(ts_edges_dist,
             dataset_name="tgbl-review",
             output_file="tgbl_review_TEA.pdf"):

    df = pd.DataFrame(ts_edges_dist)

    timestamps = np.arange(len(df))

    new = df["new"].values
    repeated = df["repeated"].values

    plt.figure(figsize=(12, 6))

    plt.bar(
        timestamps,
        repeated,
        label="Repeated",
        color="#404040",
        alpha=0.5
    )

    plt.bar(
        timestamps,
        new,
        bottom=repeated,
        label="New",
        color="#ca0020",
        alpha=0.85,
        hatch="//"
    )

    split_idx = int(0.85 * len(timestamps))

    plt.axvline(
        split_idx,
        color="blue",
        linestyle="--",
        linewidth=2
    )

    plt.title(f"TEA Plot - {dataset_name}")
    plt.xlabel("Timestamp bins")
    plt.ylabel("Number of unique edges")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

    print(f"Zapisano: {output_file}")


def main():

    dataset = PyGLinkPropPredDataset(
        name="tgbl-review",
        root="../datasets"
    )

    # 1 dzień
    interval_size = 86400

    temp_edgelist = build_temporal_edgelist(
        dataset,
        interval_size=interval_size
    )

    print('dupa1')

    ts_edges_dist, edge_frequency_dict = (
        process_edgelist_per_timestamp(temp_edgelist)
    )

    print('dupa2')

    plot_tea(
        ts_edges_dist,
        dataset_name="tgbl-review",
        output_file="tgbl_review_TEA.pdf"
    )

    mean_new = np.mean([x["new"] for x in ts_edges_dist])
    mean_total = np.mean([x["total_curr_ts"] for x in ts_edges_dist])

    print(f"Średnia liczba nowych krawędzi: {mean_new:.2f}")
    print(f"Średnia liczba krawędzi na timestamp: {mean_total:.2f}")
    print(f"New/Total ratio: {mean_new/mean_total:.4f}")


if __name__ == "__main__":
    main()