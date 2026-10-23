"""Prompt repository and text conditioning.

Repository file (`torch.save` dict):
    prompts          FloatTensor (N, K, d)   learned prompts, one row per (task, seed)
    tasks            list[str] of length N   task id of every prompt, e.g. "eurosat"
    seeds            list[int] of length N   prompt-learning seed of every prompt
    task_classnames  dict[str, list[str]]    class names describing each task (the text y(c))

Text-encoding file (`torch.save` dict): {class name: FloatTensor (D,)}, with the empty
string "" holding the null embedding used for classifier-free guidance. Class names are
encoded independently (one token per class) so that conditions are not limited by the
77-token CLIP window and class lists of different tasks can be mixed freely.
"""
import torch
from torch.utils.data import Dataset

NULL = ""


def load_repository(path):
    repo = torch.load(path, map_location="cpu")
    missing = {"prompts", "tasks", "seeds", "task_classnames"} - set(repo)
    if missing:
        raise KeyError(f"{path} is not a prompt repository (missing {sorted(missing)})")
    repo["prompts"] = repo["prompts"].float()
    assert len(repo["tasks"]) == len(repo["seeds"]) == repo["prompts"].shape[0]
    return repo


class ClassNameConditioner:
    """Turns lists of class names into zero-padded (B, max_tokens, D) cross-attention contexts."""

    def __init__(self, encodings, max_tokens=500):
        if NULL not in encodings:
            raise KeyError('text encodings need an entry for "" (the null / unconditional embedding)')
        self.encodings = {k: v.float().reshape(-1) for k, v in encodings.items()}
        self.max_tokens = max_tokens
        self.dim = self.encodings[NULL].numel()
        self._cache = {}  # encoded conditions, reused across training steps

    @classmethod
    def from_file(cls, path, max_tokens=500):
        return cls(torch.load(path, map_location="cpu"), max_tokens)

    def encode_one(self, classnames, length):
        names = list(classnames) or [NULL]
        unknown = [n for n in names if n not in self.encodings]
        if unknown:
            raise KeyError(f"no text encoding for {unknown[:5]} ({len(unknown)} missing); "
                           "add them with scripts/encode_classnames.py")
        tokens = torch.stack([self.encodings[n] for n in names])
        if len(names) < length:
            tokens = torch.cat([tokens, tokens.new_zeros(length - len(names), self.dim)])
        # every feature is L2-normalised across the token axis
        return tokens / tokens.norm(dim=0, keepdim=True).clamp_min(1e-12)

    def encode(self, batch_of_classnames, device=None):
        """An empty class list gives the null condition. All conditions of a batch are padded
        to a common length: `max_tokens`, or the longest class list if that is longer."""
        length = max(self.max_tokens, max(len(names) for names in batch_of_classnames))
        return torch.stack([self._cached(tuple(names), length, device) for names in batch_of_classnames])

    def _cached(self, names, length, device):
        key = (names, length, str(device))
        if key not in self._cache:
            self._cache[key] = self.encode_one(names, length).to(device)
        return self._cache[key]


class PromptDataset(Dataset):
    """(prompt-or-latent, task) pairs; the text condition is dropped with prob. `p_uncond`
    so the model can be sampled with classifier-free guidance."""

    def __init__(self, items, tasks, task_classnames, p_uncond=0.2):
        self.items, self.tasks, self.task_classnames, self.p_uncond = items, tasks, task_classnames, p_uncond

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        drop = torch.rand(()).item() < self.p_uncond
        return self.items[index], ([] if drop else self.task_classnames[self.tasks[index]])


def collate(batch):
    return torch.stack([item for item, _ in batch]), [names for _, names in batch]
