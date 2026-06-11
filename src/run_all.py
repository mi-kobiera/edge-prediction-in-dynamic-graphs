import itertools
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ======================
# CONFIG
# ======================

DATASETS = ["tgbl-wiki", "Enron", "uci", "UNvote", "mooc"]

LABELING_TO_MODEL = {
    "vanilla": "tgn",
    "zot": "tgn_labeling_zot",
    "de": "tgn_labeling_de",
    "drnl": "tgn_labeling_drnl",
}

SEEDS = [1, 2, 3]

MAX_WORKERS = 8
MAX_RETRIES = 2

LOG_DIR = Path("logs")
STATUS_FILE = Path("status.txt")

LOG_DIR.mkdir(exist_ok=True)


# ======================
# RESUME
# ======================

def load_done():
    if not STATUS_FILE.exists():
        return set()

    with open(STATUS_FILE, "r") as f:
        return set(line.strip() for line in f)


def mark_done(name):
    with open(STATUS_FILE, "a") as f:
        f.write(name + "\n")


# ======================
# RUN
# ======================

def run_cmd(cmd, log_file):
    with open(log_file, "w") as f:
        return subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode


def run_with_retry(cmd, log_file, name):
    for attempt in range(MAX_RETRIES + 1):
        code = run_cmd(cmd, log_file)

        if code == 0:
            return True

        print(f"[RETRY {attempt+1}] {name}")
        time.sleep(2 * (attempt + 1))

    return False


# ======================
# BUILD EXPERIMENT
# ======================

def build_experiment(dataset, labeling, seed):
    model = LABELING_TO_MODEL[labeling]
    exp_name = f"rand-{dataset}-{labeling}-seed{seed}"

    cmd = [
        "uv",
        "run",
        "src/run2.py",
        f"experiment_name={exp_name}",
        f"seed={seed}",
        f"model={model}",
        f"dataset={dataset}",
    ]

    # 👇 KLUCZOWA ZMIANA: vanilla -> brak labeling
    if labeling != "vanilla":
        cmd.append(f"labeling={labeling}")

    return exp_name, cmd


# ======================
# WORKER
# ======================

def run_experiment(dataset, labeling, seed):
    name, cmd = build_experiment(dataset, labeling, seed)
    log_file = LOG_DIR / f"{name}.log"

    if name in DONE:
        print(f"SKIP: {name}")
        return name, True

    print(f"START: {name}")

    ok = run_with_retry(cmd, log_file, name)

    if ok:
        mark_done(name)
        print(f"✓ DONE: {name}")
    else:
        print(f"✗ FAIL: {name}")

    return name, ok


# ======================
# MAIN
# ======================

def main():
    global DONE
    DONE = load_done()

    jobs = list(itertools.product(DATASETS, LABELING_TO_MODEL.keys(), SEEDS))

    print(f"Total: {len(jobs)} | Done: {len(DONE)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_experiment, d, l, s): (d, l, s)
            for d, l, s in jobs
        }

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                d, l, s = futures[future]
                print(f"ERROR {d}-{l}-{s}: {e}")


if __name__ == "__main__":
    main()