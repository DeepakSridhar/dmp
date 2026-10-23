"""Config and checkpoint helpers."""
import os

import torch
import yaml

from .autoencoder import PromptAutoencoder
from .diffusion import PromptDiffusion


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(path, model, config, step, **extra):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"config": config, "model": model.state_dict(), "step": step, **extra}, path)


def load_autoencoder(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu")
    model = PromptAutoencoder(**ckpt["config"]["autoencoder"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


class PromptAutoencoders:
    """One shared prompt autoencoder, or one per task.

    `specs` is a list of command-line values: a single checkpoint path (shared), or
    "task=path" entries giving every task its own autoencoder.
    """

    def __init__(self, specs, device="cpu"):
        if len(specs) == 1 and "=" not in specs[0]:
            self.shared, self.per_task = load_autoencoder(specs[0], device)[0], {}
        else:
            pairs = [spec.split("=", 1) for spec in specs]
            if any(len(pair) != 2 for pair in pairs):
                raise ValueError("per-task autoencoders are given as task=path")
            self.shared, self.per_task = None, {task: load_autoencoder(path, device)[0] for task, path in pairs}

    def __getitem__(self, task):
        if self.shared is not None:
            return self.shared
        if task not in self.per_task:
            raise KeyError(f"no autoencoder for task {task!r} (have {sorted(self.per_task)})")
        return self.per_task[task]

    @torch.no_grad()
    def encode(self, prompts, tasks, chunk=256):
        """Posterior mode of every prompt under its task's autoencoder."""
        latents = [None] * len(tasks)
        for task in dict.fromkeys(tasks):
            index = [i for i, t in enumerate(tasks) if t == task]
            device = next(self[task].parameters()).device
            codes = torch.cat([self[task].encode(prompts[index[i:i + chunk]].to(device)).mode().cpu()
                               for i in range(0, len(index), chunk)])
            for i, code in zip(index, codes):
                latents[i] = code
        return torch.stack(latents)

    @torch.no_grad()
    def encode_anchor(self, anchors, tasks):
        """Latent steering target for a (composite) task: the average prompt anchor encoded with
        the shared autoencoder, or the average of the task anchors each encoded with its own."""
        anchors = torch.stack([anchors[t].float() for t in tasks])
        if self.shared is not None:
            return self.encode(anchors.mean(0, keepdim=True), tasks[:1])
        return self.encode(anchors, tasks).mean(0, keepdim=True)

    @torch.no_grad()
    def decode(self, latents, task, prompt_shape):
        device = next(self[task].parameters()).device
        return self[task].decode(latents.to(device), prompt_shape)


def load_dmp(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu")
    model = PromptDiffusion(**ckpt["config"]["diffusion"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
