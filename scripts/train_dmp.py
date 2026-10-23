"""Train a Diffusion Meta-Prompt model on a prompt repository (Eq. 2).

Only the repository and the text encodings of the task descriptions are needed: no task
data and no task loss. With --autoencoder the diffusion runs in the autoencoder's latent
space; without it, directly in prompt space (K tokens = channels, d = length). Giving
--autoencoder as task=path entries encodes the prompts of each task with its own autoencoder.
"""
import argparse

from _common import infinite, pick_device
import torch
from torch.utils.data import DataLoader

from dmp.checkpoint import PromptAutoencoders, load_config, save_checkpoint
from dmp.data import ClassNameConditioner, PromptDataset, collate, load_repository
from dmp.diffusion import PromptDiffusion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--repository", required=True)
    p.add_argument("--text-encodings", required=True)
    p.add_argument("--autoencoder", nargs="+",
                   help="prompt autoencoder checkpoint, or task=path entries for one autoencoder per task "
                        "(omit to train in prompt space)")
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--steps", type=int, help="override train_dmp.steps")
    p.add_argument("--save-every", type=int, help="also save <out>.step<N> every N steps")
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    hp, cond_cfg = config["train_dmp"], config["conditioning"]
    steps = args.steps or hp["steps"]
    device = pick_device(args.device)

    repo = load_repository(args.repository)
    prompts = repo["prompts"]
    conditioner = ClassNameConditioner.from_file(args.text_encodings, cond_cfg["max_tokens"])

    if args.autoencoder:
        items = PromptAutoencoders(args.autoencoder, device).encode(prompts, repo["tasks"])
    else:
        items = prompts
    # rescale the data to unit variance; sampling divides by the same factor
    config["diffusion"]["scale_factor"] = scale_factor = 1.0 / items.std().item()
    items = items * scale_factor

    model = PromptDiffusion(**config["diffusion"]).to(device)
    assert tuple(items.shape[1:]) == model.sample_shape, (
        f"data has shape {tuple(items.shape[1:])} but the UNet expects {model.sample_shape}")
    print(f"dmp: {sum(p.numel() for p in model.unet.parameters()) / 1e6:.2f}M parameters, "
          f"sample shape {model.sample_shape}, scale_factor {scale_factor:.4f}, {len(items)} prompts")
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    dataset = PromptDataset(items, repo["tasks"], repo["task_classnames"], cond_cfg["p_uncond"])
    loader = infinite(DataLoader(dataset, batch_size=hp["batch_size"], shuffle=True, collate_fn=collate))

    model.train()
    for step in range(1, steps + 1):
        x, classnames = next(loader)
        loss = model.loss(x.to(device), conditioner.encode(classnames, device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_ema()
        if step == 1 or step % max(1, steps // 10) == 0:
            print(f"[dmp] step {step:>6}/{steps}  denoising loss {loss.item():.4f}", flush=True)
        if args.save_every and step % args.save_every == 0 and step < steps:
            save_checkpoint(f"{args.out}.step{step}", model, config, step, prompt_shape=tuple(prompts.shape[1:]))
    save_checkpoint(args.out, model, config, steps, prompt_shape=tuple(prompts.shape[1:]))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
