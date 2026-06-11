import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# =====================================================================
# CONFIG
# =====================================================================

LOG_DIR = "/Users/mi-kobiera/Projects/logs"

# Słownik: klucz to nazwa wyciągana z pliku .log, 
# wartość to ładna nazwa do wyświetlenia na wykresie w tytule.
# Kolejność: Rząd 1 (2 wykresy), Rząd 2 (2 wykresy), Rząd 3 (1 wykres na środku)
DATASETS_CONFIG = {
    "tgbl-wiki":    "tgbl-wiki",
    "Enron":  "Enron",
    "mooc":    "MOOC",
    "uci":  "UCI",
    "UNvote": "UN Vote"
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
    "axes.titlesize": 16,
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
# REGEX
# =====================================================================

test_auc_re = re.compile(r"'test/auc':\s*([0-9.]+)")
test_ap_re  = re.compile(r"'test/ap':\s*([0-9.]+)")

epoch_re = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([0-9.]+)\s+\|\s+Val AUC:\s+([0-9.]+)\s+\|\s+Val AP:\s+([0-9.]+)"
)

# =====================================================================
# PARSE LOG
# =====================================================================

def parse_log(path):
    with open(path, "r") as f:
        text = f.read()

    auc_match = test_auc_re.search(text)
    ap_match = test_ap_re.search(text)

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
# LOAD DATA
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
# PAD SEQUENCES & AGGREGATE
# =====================================================================

def pad(arr_list):
    if not arr_list:
        return np.array([])
    max_len = max(len(x) for x in arr_list)
    out = np.full((len(arr_list), max_len), np.nan)
    for i, arr in enumerate(arr_list):
        out[i, :len(arr)] = arr
    return out

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
# PLOT FUNCTION
# =====================================================================

def plot_all_datasets(metric):
    
    # Proporcje idealne dla formatu A4 w orientacji pionowej (portrait)
    fig = plt.figure(figsize=(11, 13))
    
    # Siatka 3 rzędy, 4 kolumny.
    # Pozwala to na zajęcie 2 kolumn przez każdy wykres,
    # co ułatwia idealne wyśrodkowanie ostatniego na środku.
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.5)
    
    axes = []
    # Rząd 1
    axes.append(fig.add_subplot(gs[0, 0:2]))
    axes.append(fig.add_subplot(gs[0, 2:4]))
    # Rząd 2
    axes.append(fig.add_subplot(gs[1, 0:2]))
    axes.append(fig.add_subplot(gs[1, 2:4]))
    # Rząd 3 (zajmuje kolumny od 1 do 3 - idealnie na środku)
    axes.append(fig.add_subplot(gs[2, 1:3]))

    legend_handles_labels = None

    for i, (dataset_key, display_name) in enumerate(DATASETS_CONFIG.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        # Tytuł wykresu brany ze słownika
        ax.set_title(display_name, fontweight='bold', pad=10)
        
        dataset_keys = [(d, l) for (d, l) in epoch_stats.keys() if d == dataset_key]
        
        if not dataset_keys:
            # PLACEHOLDER DLA BRAKUJĄCYCH ZBIORÓW
            ax.text(0.5, 0.5, 'Results Pending\n(Placeholder)', 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    transform=ax.transAxes, 
                    color='gray', fontsize=16, alpha=0.6)
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linestyle('--')
                spine.set_alpha(0.3)
            continue

        all_means = []
        
        for (d, label) in dataset_keys:
            data = epoch_stats[(d, label)]
            
            mean_full = data[f"{metric}_mean"]
            std_full = data[f"{metric}_std"]

            # CUT FIRST 5% OF EPOCHS
            cut = int(0.05 * len(mean_full))
            mean = mean_full[cut:]
            std = std_full[cut:]
            x = np.arange(len(mean))

            style = LABELING_STYLES.get(label, {"color": "gray", "linestyle": "-"})

            ax.plot(
                x, mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.5,  # Pogrubione linie dla lepszej czytelności po wydruku
                label=label,
                zorder=3
            )

            ax.fill_between(
                x, mean - std, mean + std,
                color=style["color"],
                alpha=0.15      # Nieco mocniejszy kolor cienia
            )
            all_means.append(mean)

        if len(all_means) > 0:
            stacked = np.concatenate(all_means)
            ymin, ymax = np.nanmin(stacked), np.nanmax(stacked)
            margin = 0.05 * (ymax - ymin + 1e-8)
            ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper() + " Score")
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.tick_params(direction="in", length=5, width=1.0)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            
        if legend_handles_labels is None:
            legend_handles_labels = ax.get_legend_handles_labels()

    # Wspólna legenda wycentrowana na samym dole wykresu
    if legend_handles_labels is not None and len(legend_handles_labels[0]) > 0:
        handles, labels = legend_handles_labels
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            frameon=False
        )

    # Margines dolny żeby legenda nie nachodziła na wykres
    fig.subplots_adjust(bottom=0.1)

    os.makedirs("figures", exist_ok=True)
    out = f"figures/All_Datasets_{metric.upper()}"
    plt.savefig(out + ".pdf")
    # plt.savefig(out + ".png", dpi=600)
    print(f"[OK] Saved {out}")
    
    plt.show()

# =====================================================================
# RUN
# =====================================================================

print("\n===== TEST RESULTS =====\n")
for (dataset, label), vals in results.items():
    auc = np.array(vals["test_auc"], dtype=float)
    ap = np.array(vals["test_ap"], dtype=float)
    print(f"{dataset} | {label}")
    print(f"  AUC: {auc.mean():.4f} ± {auc.std():.4f}")
    print(f"  AP : {ap.mean():.4f} ± {ap.std():.4f}\n")

print("Generating plots...")
plot_all_datasets("auc")
plot_all_datasets("ap")