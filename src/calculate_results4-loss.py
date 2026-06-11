import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =====================================================================
# CONFIG
# =====================================================================

LOG_DIR = "/Users/mi-kobiera/Projects/logs"

DATASETS_CONFIG = {
    "tgbl-wiki":  "tgbl-wiki",
    "Enron":      "Enron",
    "mooc":       "MOOC",
    "uci":        "UCI",
    "UNvote":     "UN Vote"
}

# =====================================================================
# IEEE STYLE
# =====================================================================

plt.style.use("default")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
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
# REGEX & PARSING
# =====================================================================

# Wyciągamy Loss (grupa 2)
epoch_re = re.compile(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([0-9.]+)\s+\|\s+Val AUC:\s+([0-9.]+)\s+\|\s+Val AP:\s+([0-9.]+)")

def parse_log(path):
    with open(path, "r") as f:
        text = f.read()

    epochs, loss = [], []
    for m in epoch_re.finditer(text):
        epochs.append(int(m.group(1)))
        loss.append(float(m.group(2))) # Zapisujemy Loss

    if len(epochs) == 0: 
        return None

    return {
        "epochs": np.array(epochs),
        "loss": np.array(loss),
    }

# =====================================================================
# LOAD DATA & AGGREGATE
# =====================================================================

results = defaultdict(lambda: defaultdict(list))

for root, _, files in os.walk(LOG_DIR):
    for file in files:
        if "log" not in file: continue
        path = os.path.join(root, file)
        parts = file.replace(".log", "").split("-")
        if len(parts) < 4: continue

        label, dataset = parts[-2], "-".join(parts[1:-2])
        parsed = parse_log(path)
        if parsed:
            results[(dataset, label)]["loss"].append(parsed["loss"])

def pad(arr_list):
    if not arr_list: return np.array([])
    max_len = max(len(x) for x in arr_list)
    out = np.full((len(arr_list), max_len), np.nan)
    for i, arr in enumerate(arr_list): out[i, :len(arr)] = arr
    return out

epoch_stats = {}
for (dataset, label), vals in results.items():
    epoch_stats[(dataset, label)] = {
        "loss_mean": np.nanmean(pad(vals["loss"]), axis=0),
        "loss_std": np.nanstd(pad(vals["loss"]), axis=0),
    }

# =====================================================================
# PLOTTING INDIVIDUAL FILES
# =====================================================================

os.makedirs("figures", exist_ok=True)

def plot_single_loss(dataset_key, display_name):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    
    dataset_keys = [(d, l) for (d, l) in epoch_stats.keys() if d == dataset_key]
    
    if not dataset_keys:
        ax.text(0.5, 0.5, 'Results Pending', ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_linestyle('--'); spine.set_alpha(0.3)
    else:
        all_means = []
        for (d, label) in dataset_keys:
            data = epoch_stats[(d, label)]
            mean_full = data["loss_mean"]
            std_full = data["loss_std"]

            # Obcięcie pierwszych 5% (szczególnie przydatne przy Loss, bo pierwsza epoka to często olbrzymi pik, który psuje skalę)
            cut = int(0.05 * len(mean_full))
            mean, std = mean_full[cut:], std_full[cut:]
            x = np.arange(len(mean))

            style = LABELING_STYLES.get(label, {"color": "gray", "linestyle": "-"})

            ax.plot(x, mean, color=style["color"], linestyle=style["linestyle"], linewidth=2.5, label=label, zorder=3)
            ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.15)
            all_means.append(mean)

        if len(all_means) > 0:
            stacked = np.concatenate(all_means)
            ymin, ymax = np.nanmin(stacked), np.nanmax(stacked)
            margin = 0.05 * (ymax - ymin + 1e-8)
            # Opcjonalnie: upewnienie się, że dolna krawędź nie spadnie poniżej zera (loss zwykle jest > 0)
            ax.set_ylim(max(0, ymin - margin), ymax + margin)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("") # Brak podpisu osi Y zgodnie z poleceniem
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.tick_params(direction="in", length=5, width=1.0)
        for spine in ax.spines.values(): spine.set_linewidth(1.2)

    out = f"figures/{dataset_key}_loss.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"[OK] Saved {out}")

def export_standalone_legend():
    fig_leg = plt.figure(figsize=(8, 0.5))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off')
    
    handles = []
    labels = []
    for label, style in LABELING_STYLES.items():
        line, = ax_leg.plot([], [], color=style["color"], linestyle=style["linestyle"], linewidth=2.5)
        handles.append(line)
        labels.append(label)
        
    fig_leg.legend(handles, labels, loc='center', ncol=4, frameon=False)
    plt.savefig("figures/legend.pdf", bbox_inches='tight')
    plt.close(fig_leg)
    print("[OK] Saved figures/legend.pdf")

# GENERATE
print("Generating loss plots...")
for d_key, d_name in DATASETS_CONFIG.items():
    plot_single_loss(d_key, d_name)

export_standalone_legend()
print("Done!")