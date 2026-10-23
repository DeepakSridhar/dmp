"""Create a small synthetic prompt repository to exercise the pipeline end to end.

Each toy task has a handful of class names with random text encodings; its "learned
prompts" are a fixed linear function of the mean class encoding plus per-seed noise, so
there is real text -> prompt structure for DMP to pick up. Also writes a fake
training-accuracy table (higher for seeds closer to the task mean) for anchor selection.
"""
import argparse
import csv
import os

import _common  # noqa: F401
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/toy/data")
    p.add_argument("--tasks", type=int, default=4)
    p.add_argument("--classes-per-task", type=int, default=6)
    p.add_argument("--seeds", type=int, default=40, help="prompts (initializations) per task")
    p.add_argument("--ctx", type=int, default=4, help="context length K")
    p.add_argument("--dim", type=int, default=16, help="prompt token dimension d")
    p.add_argument("--text-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    text_to_prompt = torch.randn(args.text_dim, args.ctx * args.dim, generator=g)

    encodings = {"": torch.randn(args.text_dim, generator=g)}
    task_classnames, prompts, tasks, seeds, rows = {}, [], [], [], []
    for t in range(args.tasks):
        task = f"task{t}"
        names = [f"{task}_class{c}" for c in range(args.classes_per_task)]
        task_classnames[task] = names
        for name in names:
            encodings[name] = torch.nn.functional.normalize(torch.randn(args.text_dim, generator=g), dim=0)
        mean = torch.stack([encodings[n] for n in names]).mean(0) @ text_to_prompt
        for s in range(1, args.seeds + 1):
            noise = 0.1 * torch.randn(args.ctx * args.dim, generator=g)
            prompts.append((mean + noise).reshape(args.ctx, args.dim))
            tasks.append(task)
            seeds.append(s)
            rows.append((task, s, round(90.0 - 20.0 * noise.norm().item(), 2)))

    torch.save({"prompts": torch.stack(prompts), "tasks": tasks, "seeds": seeds,
                "task_classnames": task_classnames}, os.path.join(args.out, "repository.pt"))
    torch.save(encodings, os.path.join(args.out, "text_encodings.pt"))
    with open(os.path.join(args.out, "train_accuracies.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "seed", "train_acc"])
        writer.writerows(rows)
    print(f"wrote {len(prompts)} prompts of shape {tuple(prompts[0].shape)} for {args.tasks} tasks to {args.out}")


if __name__ == "__main__":
    main()
