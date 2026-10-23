"""Pick the steering anchor of every task (Eq. 6): the repository prompt whose
prompt-learning seed has the highest accuracy on the task's *training* split.

The accuracy table is a CSV with columns task,seed,train_acc - numbers that already come
with the repository; no target-task or test data is involved.
"""
import argparse
import csv

import _common  # noqa: F401
import torch

from dmp.data import load_repository


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", required=True)
    p.add_argument("--accuracies", required=True, help="CSV with columns task,seed,train_acc")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    repo = load_repository(args.repository)
    index = {(task, int(seed)): i for i, (task, seed) in enumerate(zip(repo["tasks"], repo["seeds"]))}
    best = {}
    with open(args.accuracies, newline="") as f:
        for row in csv.DictReader(f):
            key, acc = (row["task"], int(row["seed"])), float(row["train_acc"])
            if key in index and acc > best.get(key[0], (None, float("-inf")))[1]:
                best[key[0]] = (key[1], acc)
    missing = sorted(set(repo["tasks"]) - set(best))
    if missing:
        raise SystemExit(f"no training accuracy found for tasks {missing}")

    torch.save({"anchors": {task: repo["prompts"][index[(task, seed)]].clone() for task, (seed, _) in best.items()},
                "seeds": {task: seed for task, (seed, _) in best.items()},
                "train_acc": {task: acc for task, (_, acc) in best.items()}}, args.out)
    for task, (seed, acc) in sorted(best.items()):
        print(f"{task:>20}: seed {seed:<4} train_acc {acc:.2f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
