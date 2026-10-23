"""Train the prompt autoencoder on a prompt repository (only needed for long prompts)."""
import argparse

from _common import infinite, pick_device
import torch
from torch.utils.data import DataLoader, TensorDataset

from dmp.autoencoder import PromptAutoencoder
from dmp.checkpoint import load_config, save_checkpoint
from dmp.data import load_repository


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--repository", required=True)
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--steps", type=int, help="override train_autoencoder.steps")
    p.add_argument("--tasks", nargs="+", help="train only on the prompts of these tasks (per-task autoencoder)")
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    hp = config["train_autoencoder"]
    steps = args.steps or hp["steps"]
    device = pick_device(args.device)

    repo = load_repository(args.repository)
    prompts = repo["prompts"]
    if args.tasks:
        keep = [i for i, task in enumerate(repo["tasks"]) if task in args.tasks]
        if not keep:
            raise SystemExit(f"no prompts for tasks {args.tasks}")
        prompts = prompts[keep]
    model = PromptAutoencoder(**config["autoencoder"]).to(device)
    print(f"autoencoder: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, "
          f"{tuple(prompts.shape[1:])} prompt -> {model.latent_shape} latent, {len(prompts)} prompts")
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], betas=(0.5, 0.9))
    loader = infinite(DataLoader(TensorDataset(prompts), batch_size=hp["batch_size"], shuffle=True))

    model.train()
    for step in range(1, steps + 1):
        (batch,) = next(loader)
        loss, logs = model.loss(batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0:
            print(f"[ae] step {step:>6}/{steps}  loss {loss.item():.4f}  rec_l1 {logs['rec_loss']:.5f}  kl {logs['kl']:.2f}", flush=True)
    save_checkpoint(args.out, model, config, steps)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
