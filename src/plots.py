import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from collections import defaultdict
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from data.loader import load_tgn_format_dataset

##############################################################################
# CONFIGURATION & LATEX AESTHETICS
##############################################################################

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 16,
    "figure.titlesize": 18
})

# DATASETS = [
#     "tgbl-wiki",       # Duży wykres (góra-lewo)
#     "tgbl-review",     # Duży wykres (góra-prawo)
#     "tgbl-coin",       # Mały wykres (dół-lewo)
#     "tgbl-comment",    # Mały wykres (dół-środek)
#     "tgbl-flight"      # Mały wykres (dół-prawo)
# ]

DATASETS = [
    "tgbl-wiki",
    "Enron",
    "mooc",
    "uci",
    "UNvote"
]

MAX_BINS = 250   # Maksymalna liczba binów dla NAJDŁUŻSZEGO zbioru
MIN_BINS = 30    # Zabezpieczenie: najkrótszy zbiór musi mieć chociaż tyle binów, by był wykresem

TEST_RATIO = 0.85
MAX_EDGES_TET = 5000   

ROOT_DIR = "../datasets"
OUTPUT_TEA = "ALL_DATASETS_TEA.pdf"
OUTPUT_TET = "ALL_DATASETS_TET.pdf"

TEA_COLOR_REP = "#A9A9A9"
TEA_COLOR_NEW = "#B2182B"

TET_COLORS = ["#FFFFFF", "#018571", "#FC8D59", "#5E3C99", "#B2182B"]
tet_cmap = ListedColormap(TET_COLORS)

##############################################################################
# HELPER FUNCTIONS
##############################################################################

def format_ticks(x, pos):
    if x >= 1e6:
        return f'{x*1e-6:g}M'
    elif x >= 1e3:
        return f'{x*1e-3:g}K'
    return str(int(x))

def format_duration(seconds):
    """Zwraca czytelny format dla rozmiaru koszyka, uwzględniając miesiące i lata."""
    days = seconds / 86400
    if days >= 365:
        return f"{days/365:.1f} Years"
    elif days >= 30:
        return f"{days/30:.1f} Months"
    elif days >= 1:
        return f"{days:.1f} Days"
    elif seconds >= 3600:
        return f"{seconds/3600:.1f} Hours"
    else:
        return f"{seconds/60:.1f} Mins"

def compute_tea(src, dst, ts, t_min, interval_size, num_bins):
    bins = defaultdict(set)
    for u, v, t in zip(src, dst, ts):
        bin_id = int((t - t_min) / interval_size)
        if bin_id >= num_bins: bin_id = num_bins - 1
        bins[bin_id].add((int(u), int(v)))

    seen_edges = set()
    rows = []

    for b in range(num_bins):
        curr = bins.get(b, set())
        repeated = curr & seen_edges
        new = curr - seen_edges
        rows.append({"new": len(new), "repeated": len(repeated)})
        seen_edges.update(curr)

    return pd.DataFrame(rows)

def compute_tet(src, dst, ts, t_min, interval_size, num_bins):
    bins = defaultdict(set)
    first_seen = {}
    last_seen = {}

    for u, v, t in zip(src, dst, ts):
        u, v = int(u), int(v)
        b = int((t - t_min) / interval_size)
        if b >= num_bins: b = num_bins - 1
        
        e = (u, v)
        bins[b].add(e)
        if e not in first_seen:
            first_seen[e] = b
        last_seen[e] = b

    split_idx = int(num_bins * TEST_RATIO)

    all_edges_sorted = sorted(first_seen.keys(), key=lambda x: first_seen[x])
    
    if len(all_edges_sorted) > MAX_EDGES_TET:
        indices = np.linspace(0, len(all_edges_sorted) - 1, MAX_EDGES_TET, dtype=int)
        edges = [all_edges_sorted[i] for i in indices]
    else:
        edges = all_edges_sorted
        
    edge_idx = {e: i for i, e in enumerate(edges)}
    
    E = len(edges)
    matrix = np.zeros((num_bins, E), dtype=np.int8)

    for ti in range(num_bins):
        for e in bins.get(ti, set()):
            if e in edge_idx:
                idx = edge_idx[e]
                fi, li = first_seen[e], last_seen[e]

                if ti <= split_idx:
                    matrix[ti, idx] = 2 if li > split_idx else 1
                else:
                    matrix[ti, idx] = 3 if fi <= split_idx else 4

    return matrix, split_idx

##############################################################################
# PRE-COMPUTE STEP: LOAD DATA & FIND PROPORTIONS
##############################################################################

print("--- STAGE 1: Loading datasets and computing timespans ---")
dataset_cache = {}
max_duration = 0

for dataset_name in DATASETS:
    print(f"Loading {dataset_name}...")

    if dataset_name == 'tgbl-wiki':
        dataset = PyGLinkPropPredDataset(name=dataset_name, root=ROOT_DIR)
        data = dataset.get_TemporalData()
    else:
        data_dir = f"/Users/mi-kobiera/Downloads/TG_network_datasets/{dataset_name}"
        data = load_tgn_format_dataset(
            data_dir=data_dir, 
            network_name=dataset_name,
            val_ratio=0.15,
            test_ratio=0.15
        )

    print(dataset_name)
    print("num_edges", data.num_edges)
    print("num_events", data.num_events)
    print("num_nodes", data.num_nodes)

    
    src, dst, ts = data.src.numpy(), data.dst.numpy(), data.t.numpy()
    
    t_min, t_max = ts.min(), ts.max()
    duration = (t_max - t_min) if (t_max > t_min) else 1
    
    if duration > max_duration:
        max_duration = duration
        
    dataset_cache[dataset_name] = (src, dst, ts, t_min, t_max, duration)

##############################################################################
# FIGURE SETUP
##############################################################################

def create_layout():
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.55, wspace=0.5)
    
    axes = [
        fig.add_subplot(gs[0, 0:3]), 
        fig.add_subplot(gs[0, 3:6]), 
        fig.add_subplot(gs[1, 0:2]), 
        fig.add_subplot(gs[1, 2:4]), 
        fig.add_subplot(gs[1, 4:6])  
    ]
    return fig, axes

fig_tea, axes_tea = create_layout()
fig_tet, axes_tet = create_layout()

##############################################################################
# MAIN EXECUTION (STAGE 2)
##############################################################################

print("\n--- STAGE 2: Computing matrices and plotting ---")

for i, dataset_name in enumerate(DATASETS):
    src, dst, ts, t_min, t_max, duration = dataset_cache[dataset_name]
    
    # Krok Proporcjonalny: obliczenie liczby binów wg najdłuższego zbioru
    proportion = duration / max_duration
    num_bins = int(np.ceil(proportion * MAX_BINS))
    
    # Zabezpieczenie by najmniejsze zbiory nie zniknęły
    num_bins = max(MIN_BINS, min(num_bins, MAX_BINS)) 
    
    interval_size = duration / num_bins
    time_label = format_duration(interval_size)
    
    print(f"[{dataset_name}] Bins: {num_bins} ({proportion*100:.1f}% of max), 1 Bin ≈ {time_label}")
    plot_title = f"{dataset_name}"
    
    # ----------------------------------------------------
    # 1. TEA PLOT
    # ----------------------------------------------------
    df_tea = compute_tea(src, dst, ts, t_min, interval_size, num_bins)
    x = np.arange(len(df_tea))
    new = df_tea["new"].values
    repeated = df_tea["repeated"].values
    split_idx_tea = int(TEST_RATIO * num_bins)
    
    ax_tea = axes_tea[i]
    ax_tea.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    ax_tea.bar(x, repeated, width=1.0, color=TEA_COLOR_REP, alpha=0.7, zorder=3)
    ax_tea.bar(x, new, bottom=repeated, width=1.0, color=TEA_COLOR_NEW, alpha=0.9, zorder=3)
    ax_tea.axvline(x=split_idx_tea, color="black", linestyle="--", linewidth=2)
    
    ax_tea.set_title(plot_title, fontweight='semibold')
    ax_tea.set_xlabel("Timestamp")
    ax_tea.yaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax_tea.margins(x=0.01)
    
    if i in [0, 2]:
        ax_tea.set_ylabel("Number of edges")

    # ----------------------------------------------------
    # 2. TET PLOT
    # ----------------------------------------------------
    matrix_tet, split_idx_tet = compute_tet(src, dst, ts, t_min, interval_size, num_bins)
    
    ax_tet = axes_tet[i]
    ax_tet.imshow(
        matrix_tet, 
        cmap=tet_cmap, 
        aspect='auto', 
        vmin=0, vmax=4,
        interpolation='nearest'
    )
    
    ax_tet.axhline(split_idx_tet, color="black", linestyle="--", linewidth=2)
    ax_tet.set_title(plot_title, fontweight='semibold')
    ax_tet.set_xlabel("Percentage of observed edges")
    ax_tet.set_xticks([])
    
    if i in [0, 2]:
        ax_tet.set_ylabel("Timestamp")
    else:
        ax_tet.set_yticks([]) 

# ----------------------------------------------------
# 3. GLOBALNE LEGENDY
# ----------------------------------------------------
tea_legend_patches = [
    mpatches.Patch(color=TEA_COLOR_REP, alpha=0.7, label='Repeated Edges'),
    mpatches.Patch(color=TEA_COLOR_NEW, alpha=0.9, label='New Edges'),
    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Train/Test Split')
]
fig_tea.legend(handles=tea_legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

tet_legend_patches = [
    mpatches.Patch(color=TET_COLORS[1], label='Train ∖ Test'),
    mpatches.Patch(color=TET_COLORS[2], label='Train ∩ Test'),
    # mpatches.Patch(color=TET_COLORS[3], label='Transductive'),
    mpatches.Patch(color=TET_COLORS[4], label='Train'),
    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Train/Test Split')
]
fig_tet.legend(handles=tet_legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)

print("\nSaving figures...")
fig_tea.savefig(OUTPUT_TEA, format='pdf', bbox_inches="tight", dpi=300)
fig_tet.savefig(OUTPUT_TET, format='pdf', bbox_inches="tight", dpi=300)

print(f"Done! Saved: {OUTPUT_TEA} and {OUTPUT_TET}")