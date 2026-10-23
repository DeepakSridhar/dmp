"""Convert pytorch-lightning checkpoints and prompt dictionaries into the formats used here
(pytorch-lightning does not need to be installed).

  dmp          lightning DMP checkpoint          -> DMP checkpoint (UNet + EMA weights)
  autoencoder  lightning autoencoder checkpoint  -> autoencoder checkpoint
  repository   {"seed<S>-<task>_classes": prompt} dict + class-name JSON -> repository file
"""
import argparse
import json
import pickle
import re
import types

import _common  # noqa: F401
import torch

from dmp.autoencoder import PromptAutoencoder
from dmp.checkpoint import load_config, save_checkpoint
from dmp.diffusion import PromptDiffusion


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    """Lightning checkpoints pickle callback/config objects; replace them by inert stubs."""

    def find_class(self, module, name):
        if module.split(".")[0] in ("pytorch_lightning", "lightning", "omegaconf"):
            return type(name, (_Stub,), {})
        return super().find_class(module, name)


_pickle = types.ModuleType("pickle")
_pickle.Unpickler, _pickle.load, _pickle.__name__ = _Unpickler, pickle.load, "pickle"


def load_lightning_state(path):
    return torch.load(path, map_location="cpu", pickle_module=_pickle)["state_dict"]


def convert_dmp(args):
    config = load_config(args.config)
    config["diffusion"]["scale_factor"] = args.scale_factor
    state = load_lightning_state(args.ckpt)
    prefix = next(p for p in ("model.diffusion_model.ddpm_list.0.", "model.diffusion_model.")
                  if any(k.startswith(p) for k in state))
    model = PromptDiffusion(**config["diffusion"])
    unet = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    model.unet.load_state_dict(unet)
    # lightning stores EMA weights under the parameter name with the dots removed
    ema = {name: state.get("model_ema." + (prefix[len("model."):] + name).replace(".", "")) for name in unet}
    if all(v is not None for v in ema.values()):
        model.ema_unet.load_state_dict(ema)
        print(f"loaded {len(unet)} UNet tensors and their EMA copies")
    else:
        model.ema_unet.load_state_dict(unet)
        print(f"loaded {len(unet)} UNet tensors (no EMA weights found; using the raw weights)")
    save_checkpoint(args.out, model, config, 0, prompt_shape=tuple(args.prompt_shape))


def convert_autoencoder(args):
    config = load_config(args.config)
    state = load_lightning_state(args.ckpt)
    keep = ("encoder.", "decoder.", "quant_conv.", "post_quant_conv.")
    new = {k: v for k, v in state.items() if k.startswith(keep)}
    new["logvar"] = state.get("loss.logvar", torch.zeros(()))
    model = PromptAutoencoder(**config["autoencoder"])
    model.load_state_dict(new)
    print(f"loaded {len(new)} autoencoder tensors")
    save_checkpoint(args.out, model, config, 0)


def convert_repository(args):
    entries = torch.load(args.prompts, map_location="cpu")
    with open(args.classnames) as f:
        task_classnames = json.load(f)
    prompts, tasks, seeds = [], [], []
    for key, value in entries.items():
        match = re.fullmatch(r"seed(\d+)-(.+)_classes", key)
        if not match or match.group(2) not in task_classnames:
            raise SystemExit(f"cannot place repository entry {key!r} (known tasks: {sorted(task_classnames)})")
        prompts.append(value.float())
        seeds.append(int(match.group(1)))
        tasks.append(match.group(2))
    torch.save({"prompts": torch.stack(prompts), "tasks": tasks, "seeds": seeds,
                "task_classnames": {t: task_classnames[t] for t in sorted(set(tasks))}}, args.out)
    print(f"{len(prompts)} prompts of shape {tuple(prompts[0].shape)} for {len(set(tasks))} tasks")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("dmp")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--config", required=True, help="config whose `diffusion` section matches the checkpoint")
    s.add_argument("--scale-factor", type=float, required=True,
                   help="`scale_factor` of the training config")
    s.add_argument("--prompt-shape", type=int, nargs="+", default=[4, 512], help="K d of the decoded prompts")
    s.add_argument("--out", required=True)
    s.set_defaults(func=convert_dmp)

    s = sub.add_parser("autoencoder")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--config", required=True, help="config whose `autoencoder` section matches the checkpoint")
    s.add_argument("--out", required=True)
    s.set_defaults(func=convert_autoencoder)

    s = sub.add_parser("repository")
    s.add_argument("--prompts", required=True)
    s.add_argument("--classnames", required=True, help='JSON {"task": ["class name", ...]}')
    s.add_argument("--out", required=True)
    s.set_defaults(func=convert_repository)

    args = p.parse_args()
    args.func(args)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
