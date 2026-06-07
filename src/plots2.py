import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# DATA PATH CONFIGURATION
# =====================================================================

DATA_FILES = {
    "Random": {
        "AP": {
            "No Labeling": "/Users/mi-kobiera/Downloads/ap-wikipedia-random-vanilla.csv",
            "Zero-One-Two": "/Users/mi-kobiera/Downloads/ap-wikipedia-random-zot.csv",
            "DRNL": "/Users/mi-kobiera/Downloads/ap-wikipedia-random-drnl.csv",
            "DE": "/Users/mi-kobiera/Downloads/ap-wikipedia-random-de.csv"
        },
        "AUC": {
            "No Labeling": "/Users/mi-kobiera/Downloads/auc-wikipedia-random-vanilla.csv",
            "Zero-One-Two": "/Users/mi-kobiera/Downloads/auc-wikipedia-random-zot.csv",
            "DRNL": "/Users/mi-kobiera/Downloads/auc-wikipedia-random-drnl.csv",
            "DE": "/Users/mi-kobiera/Downloads/auc-wikipedia-random-de.csv"
        }
    },
    "Historical": {
        "AP": {
            "No Labeling": "/Users/mi-kobiera/Downloads/ap-wikipedia-hist-vanilla.csv",
            "Zero-One-Two": "/Users/mi-kobiera/Downloads/ap-wikipedia-hist-zot.csv",
            "DRNL": "/Users/mi-kobiera/Downloads/ap-wikipedia-hist-drnl.csv",
            "DE": "/Users/mi-kobiera/Downloads/ap-wikipedia-hist-de.csv"
        },
        "AUC": {
            "No Labeling": "/Users/mi-kobiera/Downloads/auc-wikipedia-hist-vanilla.csv",
            "Zero-One-Two": "/Users/mi-kobiera/Downloads/auc-wikipedia-hist-zot.csv",
            "DRNL": "/Users/mi-kobiera/Downloads/auc-wikipedia-hist-drnl.csv",
            "DE": "/Users/mi-kobiera/Downloads/auc-wikipedia-hist-de.csv"
        }
    }
}

# =====================================================================
# IEEE STYLE SETTINGS
# =====================================================================

plt.style.use("default")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],

    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,

    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# =====================================================================
# CLEAN COLOR PALETTE (COLORBLIND SAFE)
# =====================================================================

LABELING_STYLES = {
    "No Labeling":   {"color": "#000000", "linestyle": "-"},
    "Zero-One-Two":  {"color": "#0072B2", "linestyle": "--"},
    "DRNL":          {"color": "#D55E00", "linestyle": "-."},
    "DE":            {"color": "#009E73", "linestyle": ":"},
}

# =====================================================================
# PLOT FUNCTION
# =====================================================================

def plot_sampling_figure(sampling_name, metrics_dict):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    handles, labels = [], []

    for ax, (metric_name, label_files) in zip(axes, metrics_dict.items()):

        all_values = []

        # -------------------------------------------------------------
        # Robust Y range (percentiles)
        # -------------------------------------------------------------
        for filepath in label_files.values():
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                all_values.extend(df["Value"].values)

        all_values = np.array(all_values)

        y_min = np.percentile(all_values, 5)
        y_max = np.percentile(all_values, 99)
        padding = (y_max - y_min) * 0.08

        ax.set_ylim(y_min - padding, y_max + padding)

        # -------------------------------------------------------------
        # Plot each series
        # -------------------------------------------------------------
        for label, filepath in label_files.items():

            if not os.path.exists(filepath):
                print(f"[WARNING] Missing file: {filepath}")
                continue

            style = LABELING_STYLES[label]
            df = pd.read_csv(filepath)

            ax.plot(
                df["Step"],
                df["Value"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.6,
                label=label,
                zorder=3
            )

            if metric_name == "AP":
                handles.append(ax.lines[-1])
                labels.append(label)

        # -------------------------------------------------------------
        # Axis styling (IEEE clean)
        # -------------------------------------------------------------
        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")

        ax.grid(True, alpha=0.12, linewidth=0.5)

        ax.tick_params(direction="in", length=4, width=0.8)

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        ax.set_facecolor("white")

    # =================================================================
    # LEGEND (clean, bottom-centered)
    # =================================================================

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        frameon=False
    )

    fig.subplots_adjust(
        bottom=0.20,
        top=0.88,
        wspace=0.25
    )

    # =================================================================
    # SAVE
    # =================================================================

    output_name = f"Figure_{sampling_name}"

    plt.savefig(f"{output_name}.pdf")
    plt.savefig(f"{output_name}.png")

    print(f"[OK] Saved: {output_name}.pdf / .png")

    plt.show()


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    for sampling_method, metrics in DATA_FILES.items():

        if "AP" in metrics and "AUC" in metrics:
            print(f"Generating: {sampling_method}")
            plot_sampling_figure(sampling_method, metrics)