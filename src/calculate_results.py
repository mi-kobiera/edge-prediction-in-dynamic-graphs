import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =====================================================================
# CONFIG
# =====================================================================

LOG_DIR = "/Users/mi-kobiera/Projects/logs"

# =====================================================================
# IEEE STYLE
# =====================================================================

plt.style.use("default")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],

    "font.size": 14,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 16,

    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

LABELING_STYLES = {
    "vanilla": {"color": "#000000", "linestyle": "-"},
    "zot":     {"color": "#0072B2", "linestyle": "--"},
    "drnl":    {"color": "#D55E00", "linestyle": "-."},
    "de":      {"color": "#009E73", "linestyle": ":"},
}

# =====================================================================
# REGEX
# =====================================================================

test_auc_re = re.compile(r"'test/auc':\s*([0-9.]+)")
test_ap_re  = re.compile(r"'test/ap':\s*([0-9.]+)")

epoch_re = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([0-9.]+)\s+\|\s+Val AUC:\s+([0-9.]+)\s+\|\s+Val AP:\s+([0-9.]+)"
)

# =====================================================================
# PARSE LOG (SAFE)
# =====================================================================

def parse_log(path):
    with open(path, "r") as f:
        text = f.read()

    auc_match = test_auc_re.search(text)
    ap_match = test_ap_re.search(text)

    # -------------------------------------------------------------
    # SAFE GUARD (NO CRASH)
    # -------------------------------------------------------------
    if auc_match is None or ap_match is None:
        print(f"[WARNING] Skipping broken log (missing test metrics): {path}")
        return None

    epochs, val_auc, val_ap = [], [], []

    for m in epoch_re.finditer(text):
        epochs.append(int(m.group(1)))
        val_auc.append(float(m.group(3)))
        val_ap.append(float(m.group(4)))

    if len(epochs) == 0:
        print(f"[WARNING] Skipping log (no epochs found): {path}")
        return None

    return {
        "test_auc": float(auc_match.group(1)),
        "test_ap": float(ap_match.group(1)),
        "epochs": np.array(epochs),
        "val_auc": np.array(val_auc),
        "val_ap": np.array(val_ap),
    }

# =====================================================================
# LOAD DATA (ROBUST DATASET PARSING)
# =====================================================================

results = defaultdict(lambda: defaultdict(list))

for root, _, files in os.walk(LOG_DIR):
    for file in files:
        if "log" not in file:
            continue

        path = os.path.join(root, file)
        name = file.replace(".log", "")
        parts = name.split("-")

        if len(parts) < 4:
            continue

        # ---------------------------------------------------------
        # FIX: supports tgbl-wiki and any multi-hyphen dataset
        # ---------------------------------------------------------
        seed = parts[-1]
        label = parts[-2]
        dataset = "-".join(parts[1:-2])

        parsed = parse_log(path)
        if parsed is None:
            continue

        key = (dataset, label)

        results[key]["test_auc"].append(parsed["test_auc"])
        results[key]["test_ap"].append(parsed["test_ap"])

        results[key]["epochs"].append(parsed["epochs"])
        results[key]["val_auc"].append(parsed["val_auc"])
        results[key]["val_ap"].append(parsed["val_ap"])

# =====================================================================
# TEST RESULTS (MEAN ± STD)
# =====================================================================

print("\n===== TEST RESULTS =====\n")

for (dataset, label), vals in results.items():

    auc = np.array(vals["test_auc"], dtype=float)
    ap = np.array(vals["test_ap"], dtype=float)

    print(f"{dataset} | {label}")
    print(f"  AUC: {auc.mean():.4f} ± {auc.std():.4f}")
    print(f"  AP : {ap.mean():.4f} ± {ap.std():.4f}\n")

# =====================================================================
# PAD SEQUENCES
# =====================================================================

def pad(arr_list):
    max_len = max(len(x) for x in arr_list)
    out = np.full((len(arr_list), max_len), np.nan)

    for i, arr in enumerate(arr_list):
        out[i, :len(arr)] = arr

    return out

# =====================================================================
# AGGREGATE CURVES
# =====================================================================

epoch_stats = {}

for (dataset, label), vals in results.items():

    auc_mat = pad(vals["val_auc"])
    ap_mat = pad(vals["val_ap"])

    epoch_stats[(dataset, label)] = {
        "auc_mean": np.nanmean(auc_mat, axis=0),
        "auc_std": np.nanstd(auc_mat, axis=0),

        "ap_mean": np.nanmean(ap_mat, axis=0),
        "ap_std": np.nanstd(ap_mat, axis=0),
    }

# =====================================================================
# PLOT
# =====================================================================

# def plot_dataset(dataset_name):

#     fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

#     for ax, metric in zip(axes, ["auc", "ap"]):

#         for (dataset, label), data in epoch_stats.items():

#             if dataset != dataset_name:
#                 continue

#             style = LABELING_STYLES.get(label, {"color": "gray", "linestyle": "-"})

#             x = np.arange(len(data[f"{metric}_mean"]))
#             mean = data[f"{metric}_mean"]
#             std = data[f"{metric}_std"]

#             ax.plot(
#                 x,
#                 mean,
#                 color=style["color"],
#                 linestyle=style["linestyle"],
#                 linewidth=1.8,
#                 label=label,
#                 zorder=3
#             )

#             ax.fill_between(
#                 x,
#                 mean - std,
#                 mean + std,
#                 color=style["color"],
#                 alpha=0.18
#             )

#         ax.set_title(metric.upper())
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel("Score")

#         ax.grid(True, alpha=0.12, linewidth=0.5)
#         ax.tick_params(direction="in", length=4, width=0.8)

#         for spine in ax.spines.values():
#             spine.set_linewidth(1.0)

#     handles, labels = axes[0].get_legend_handles_labels()

#     fig.legend(
#         handles,
#         labels,
#         loc="lower center",
#         bbox_to_anchor=(0.5, -0.05),
#         ncol=4,
#         frameon=False
#     )

#     fig.subplots_adjust(bottom=0.20, top=0.88, wspace=0.25)

#     out = f"figures/Figure_{dataset_name}"
#     plt.savefig(out + ".pdf")
#     plt.savefig(out + ".png")

#     print(f"[OK] Saved {out}")

#     plt.show()

def plot_dataset(dataset_name):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, metric in zip(axes, ["auc", "ap"]):

        all_means = []

        for (dataset, label), data in epoch_stats.items():

            if dataset != dataset_name:
                continue

            mean_full = data[f"{metric}_mean"]
            std_full = data[f"{metric}_std"]

            # ---------------------------------------------------------
            # CUT FIRST 5% OF EPOCHS
            # ---------------------------------------------------------
            cut = int(0.05 * len(mean_full))
            mean = mean_full[cut:]
            std = std_full[cut:]

            x = np.arange(len(mean))

            style = LABELING_STYLES.get(label, {"color": "gray", "linestyle": "-"})

            ax.plot(
                x,
                mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2,   # slightly thicker
                label=label,
                zorder=3
            )

            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=style["color"],
                alpha=0.12   # lighter band = clearer lines
            )

            all_means.append(mean)

        # ---------------------------------------------------------
        # DYNAMIC Y-LIMITS (ZOOM INTO SIGNAL)
        # ---------------------------------------------------------
        if len(all_means) > 0:
            stacked = np.concatenate(all_means)
            ymin, ymax = np.nanmin(stacked), np.nanmax(stacked)

            margin = 0.05 * (ymax - ymin + 1e-8)
            ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_title(metric.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")

        ax.grid(True, alpha=0.12, linewidth=0.5)
        ax.tick_params(direction="in", length=4, width=0.8)

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        frameon=False
    )

    fig.subplots_adjust(bottom=0.20, top=0.88, wspace=0.25)

    out = f"figures/rand-{dataset_name}"
    plt.savefig(out + ".pdf")
    # plt.savefig(out + ".png", dpi=600)

    print(f"[OK] Saved {out}")
    plt.show()

# =====================================================================
# RUN
# =====================================================================

# datasets = sorted(set(k[0] for k in results.keys()))
datasets = set(k[0] for k in results.keys())

for d in datasets:
    print(f"Plotting: {d}")
    plot_dataset(d)