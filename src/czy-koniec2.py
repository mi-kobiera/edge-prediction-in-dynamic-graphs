from pathlib import Path
import re
from collections import defaultdict

LOG_DIR = Path("logs_hist")

# rand-Enron-de-seed2
name_pattern = re.compile(
    r"^hist-(?P<dataset>.+)-(?P<labeling>.+)-seed(?P<seed>\d+)$"
)

epoch_pattern = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\|\s+Loss: [\d.]+\s+\|\s+Val AUC: (?P<auc>[\d.]+)\s+\|\s+Val AP: (?P<ap>[\d.]+)"
)

EARLY_STOP_TEXT = "Early stopping triggered"

EXPECTED_SEEDS = {1, 2, 3}

# -------------------------
# STORAGE
# -------------------------
completed = defaultdict(set)

last_epoch = defaultdict(lambda: None)
best_metrics = defaultdict(lambda: {"auc": None, "ap": None})
early_stopping = defaultdict(lambda: False)

# -------------------------
# PARSING LOGS
# -------------------------
for log_file in LOG_DIR.iterdir():
    if not log_file.is_file():
        continue

    match = name_pattern.match(log_file.stem)
    if not match:
        continue

    dataset = match.group("dataset")
    labeling = match.group("labeling")
    seed = int(match.group("seed"))

    exp_key = (dataset, labeling)
    run_key = (dataset, labeling, seed)

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # finished detection
        if "Test results:" in content:
            completed[exp_key].add(seed)

        # early stopping
        if EARLY_STOP_TEXT in content:
            early_stopping[run_key] = True

        # epochs parsing
        for m in epoch_pattern.finditer(content):
            epoch = int(m.group("epoch"))
            auc = float(m.group("auc"))
            ap = float(m.group("ap"))

            last_epoch[run_key] = epoch

            # best by AP
            if (best_metrics[run_key]["ap"] is None or
                ap > best_metrics[run_key]["ap"]):
                best_metrics[run_key]["ap"] = ap
                best_metrics[run_key]["auc"] = auc

    except Exception as e:
        print(f"Error reading {log_file}: {e}")

# -------------------------
# ALL EXPERIMENTS
# -------------------------
all_experiments = set()

for log_file in LOG_DIR.iterdir():
    if not log_file.is_file():
        continue

    match = name_pattern.match(log_file.stem)
    if match:
        all_experiments.add(
            (match.group("dataset"), match.group("labeling"))
        )

# -------------------------
# SPLIT FINISHED / UNFINISHED
# -------------------------
finished = []
unfinished = []

for exp in sorted(all_experiments):
    done = completed[exp]

    if done == EXPECTED_SEEDS:
        finished.append(exp)
    else:
        missing = sorted(EXPECTED_SEEDS - done)
        unfinished.append((exp, missing))

# -------------------------
# PRINT RESULTS
# -------------------------
print("=== FINISHED ===")
for d, l in finished:
    print(f"✓ {d} / {l}")

print(f"\nFinished: {len(finished)}")

print("\n=== UNFINISHED (detailed) ===")

for (dataset, labeling), missing in unfinished:
    print(f"\n✗ {dataset} / {labeling}")
    print(f"   missing seeds: {missing}")

    for seed in missing:
        run_key = (dataset, labeling, seed)

        le = last_epoch[run_key]
        bm = best_metrics[run_key]
        es = early_stopping[run_key]

        print(f"   - seed {seed}:")

        print(f"     last epoch: {le if le is not None else 'N/A'}")

        if bm["ap"] is not None:
            print(f"     best Val AP: {bm['ap']:.4f}")
            print(f"     best Val AUC: {bm['auc']:.4f}")
        else:
            print(f"     best Val AP: N/A")
            print(f"     best Val AUC: N/A")

        print(f"     early stopping: {'yes' if es else 'no'}")

print(f"\nUnfinished: {len(unfinished)}")