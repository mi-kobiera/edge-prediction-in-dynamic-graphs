import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


DATASET = "tgbl-wiki"
INTERVAL = 86400
TEST_RATIO = 0.85
OUTPUT_FILE = f"{DATASET}_TEA.pdf"


dataset = PyGLinkPropPredDataset(name=DATASET, root="../datasets")
data = dataset.get_TemporalData()

src, dst, ts = data.src.numpy(), data.dst.numpy(), data.t.numpy()


##############################################################################
# BUILD STRUCTURES
##############################################################################

bins = defaultdict(set)
first_seen = {}
last_seen = {}

for u, v, t in zip(src, dst, ts):

    u, v = int(u), int(v)
    b = int(t // INTERVAL)
    e = (u, v)

    bins[b].add(e)

    if e not in first_seen:
        first_seen[e] = b
    last_seen[e] = b


time_bins = sorted(bins.keys())
split = int(len(time_bins) * TEST_RATIO)


##############################################################################
# EDGE LIST
##############################################################################

edges = list(first_seen.keys())
edge_idx = {e: i for i, e in enumerate(edges)}

T = len(time_bins)
E = len(edges)

mat = np.zeros((T, E), dtype=np.int8)


##############################################################################
# BUILD PRESENCE MATRIX (FAST BUT CORRECT)
##############################################################################

for ti, b in enumerate(time_bins):
    for e in bins[b]:
        mat[ti, edge_idx[e]] = 1


##############################################################################
# CLASSIFICATION (TET LOGIC)
##############################################################################

only_train = np.zeros((T, E))
train_test = np.zeros((T, E))
transductive = np.zeros((T, E))
inductive = np.zeros((T, E))

for ti, b in enumerate(time_bins):

    for e in bins[b]:

        fi = first_seen[e]
        li = last_seen[e]

        if ti <= split:

            if li > split:
                train_test[ti, edge_idx[e]] = 1
            else:
                only_train[ti, edge_idx[e]] = 1

        else:

            if fi <= split:
                transductive[ti, edge_idx[e]] = 1
            else:
                inductive[ti, edge_idx[e]] = 1


##############################################################################
# PLOT (TRUE TET STYLE)
##############################################################################

fig, ax = plt.subplots(figsize=(10, 5))

matrix = only_train + 2 * train_test + 3 * transductive + 4 * inductive

colors = [
    "white",
    "#018571",
    "#fc8d59",
    "#fc8d59",
    "#b2182b"
]

sns.heatmap(
    matrix,
    cmap=sns.color_palette(colors, as_cmap=True),
    cbar=False
)

plt.axhline(split * len(time_bins), color="black", linestyle="--")

plt.xlabel("Edges")
plt.ylabel("Time")
plt.title("TET - correct version")

plt.savefig(OUTPUT_FILE, bbox_inches="tight")
plt.show()