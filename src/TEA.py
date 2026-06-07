from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


##############################################################################
# CONFIG
##############################################################################

DATASET_NAME = "tgbl-flight"
INTERVAL_SIZE = 86400  # 1 day (change to 7*86400 for smoother plots)

OUTPUT_FILE = f"{DATASET_NAME}_TEA.pdf"


##############################################################################
# LOAD DATASET
##############################################################################

print("Loading dataset...")

dataset = PyGLinkPropPredDataset(
    name=DATASET_NAME,
    root="../datasets"
)

data = dataset.get_TemporalData()

src = data.src.numpy()
dst = data.dst.numpy()
ts = data.t.numpy()

print(f"Total edges: {len(src):,}")


##############################################################################
# BUILD TIME BUCKETS (FAST)
##############################################################################

print("Building time bins...")

bins = defaultdict(set)

for u, v, t in zip(src, dst, ts):
    bin_id = int(t // INTERVAL_SIZE)
    bins[bin_id].add((int(u), int(v)))

timestamps = sorted(bins.keys())

print(f"Number of time bins: {len(timestamps):,}")


##############################################################################
# TEA COMPUTATION (O(E))
##############################################################################

print("Computing TEA stats...")

seen_edges = set()

rows = []

for i, t_bin in enumerate(timestamps):

    curr = bins[t_bin]

    repeated = curr & seen_edges
    new = curr - seen_edges

    rows.append({
        "ts": t_bin,
        "new": len(new),
        "repeated": len(repeated),
        "total_curr_ts": len(curr),
    })

    seen_edges.update(curr)

    if i % 100 == 0:
        print(f"{i}/{len(timestamps)} bins processed")


df = pd.DataFrame(rows)


##############################################################################
# STATS
##############################################################################

avg_new = df["new"].mean()
avg_total = df["total_curr_ts"].mean()

print("\n===== STATS =====")
print(f"Avg new edges per bin   : {avg_new:.2f}")
print(f"Avg total edges per bin : {avg_total:.2f}")
print(f"New/Total ratio         : {avg_new / avg_total:.4f}")


##############################################################################
# PLOT (ORIGINAL TEA STYLE)
##############################################################################

print("Plotting...")

x = np.arange(len(df))

new = df["new"].values
repeated = df["repeated"].values

plt.figure(figsize=(10, 8))
plt.subplots_adjust(bottom=0.2, left=0.2)

font_size = 20
ticks_font_size = 18

# --- TEA ORIGINAL COLORS ---
plt.bar(
    x,
    repeated,
    label="Repeated",
    color="#404040",
    alpha=0.4
)

plt.bar(
    x,
    new,
    bottom=repeated,
    label="New",
    color="#ca0020",
    alpha=0.8,
    hatch="//"
)

# --- SPLIT LINE (85%) ---
split_idx = int(0.85 * len(x))

plt.axvline(
    x=split_idx,
    color="blue",
    linestyle="--",
    linewidth=2
)

plt.text(
    split_idx,
    max(new + repeated) * 0.02,
    "x",
    va="center",
    ha="center",
    fontsize=font_size,
    fontweight="heavy",
    color="blue"
)


##############################################################################
# AXIS FORMATTING (same style as original TEA code)
##############################################################################

plt.xlabel("Timestamp", fontsize=font_size)
plt.ylabel("Number of edges", fontsize=font_size)

plt.xticks(fontsize=ticks_font_size)
plt.yticks(fontsize=ticks_font_size)

plt.margins(x=0)
plt.legend()

plt.tight_layout()

plt.savefig(OUTPUT_FILE, bbox_inches="tight")
plt.show()

print(f"Saved to: {OUTPUT_FILE}")