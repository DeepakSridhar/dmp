"""Sample prompts from a trained DMP for one task, a composite of tasks, or any class list.

Examples
  # one task, with test-time steering (Eq. 7, lambda = 0.2)
  python scripts/sample.py --dmp dmp.pt --autoencoder ae.pt --text-encodings enc.pt \
      --repository repo.pt --tasks eurosat --anchors anchors.pt --out eurosat.pt

  # composite task: one condition holding the class names of both tasks; the anchor is
  # the average of the two task anchors
  python scripts/sample.py ... --tasks eurosat oxford_flowers --anchors anchors.pt --out pair.pt

  # concept composition / negation with Eq. 5 (one guidance weight per task)
  python scripts/sample.py ... --tasks eurosat oxford_flowers --compose --scale 4.5 -2.0 --out mix.pt
"""
import argparse

from _common import pick_device
import torch

from dmp.checkpoint import PromptAutoencoders, load_dmp
from dmp.data import ClassNameConditioner, load_repository
from dmp.sampler import DDIMSampler


def write_coop_checkpoint(template, prompt, out):
    """Copy a CoOp prompt-learner checkpoint, replacing its context vectors with `prompt`."""
    ckpt = torch.load(template, map_location="cpu")
    state = ckpt["state_dict"]
    key = next(k for k in state if "ctx" in k)
    state[key] = prompt.reshape(state[key].shape).to(state[key].dtype)
    torch.save({"state_dict": state, "epoch": ckpt.get("epoch", 0)}, out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dmp", required=True, help="DMP checkpoint")
    p.add_argument("--autoencoder", nargs="+",
                   help="prompt autoencoder checkpoint, or task=path entries for per-task autoencoders, "
                        "if the DMP was trained with them")
    p.add_argument("--decode-task", help="with per-task autoencoders: whose decoder maps the sample back "
                                         "to a prompt (default: the first of --tasks)")
    p.add_argument("--text-encodings", required=True)
    p.add_argument("--repository", help="repository providing the class names of --tasks")
    p.add_argument("--tasks", nargs="+", default=[], help="task ids to condition on")
    p.add_argument("--classnames", nargs="+", help="condition on these class names instead of --tasks")
    p.add_argument("--compose", action="store_true",
                   help="compose the tasks with Eq. 5 instead of concatenating their class names")
    p.add_argument("--scale", type=float, nargs="+", default=[4.5],
                   help="guidance scale; with --compose one value per task (negative = negation)")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anchors", help="anchors from select_anchors.py; enables test-time steering")
    p.add_argument("--steering", type=float, default=0.2, help="steering strength lambda (Eq. 7)")
    p.add_argument("--out", required=True, help="output .pt with the sampled prompts")
    p.add_argument("--coop-template", help="CoOp prompt_learner checkpoint to copy the prompt into")
    p.add_argument("--coop-out", help="where to write the CoOp checkpoint holding the first sample")
    p.add_argument("--device")
    args = p.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_dmp(args.dmp, device)
    prompt_shape = tuple(ckpt["prompt_shape"])
    autoencoders = PromptAutoencoders(args.autoencoder, device) if args.autoencoder else None
    decode_task = args.decode_task or (args.tasks[0] if args.tasks else None)
    if autoencoders and autoencoders.shared is None and decode_task is None:
        raise SystemExit("per-task autoencoders need --decode-task (or --tasks)")
    conditioner = ClassNameConditioner.from_file(args.text_encodings, ckpt["config"]["conditioning"]["max_tokens"])
    scale_factor = model.scale_factor.item()

    # ---- text conditions
    if args.classnames:
        conditions = [args.classnames]
    else:
        if not (args.tasks and args.repository):
            raise SystemExit("give --classnames, or --tasks together with --repository")
        task_classnames = load_repository(args.repository)["task_classnames"]
        per_task = [task_classnames[t] for t in args.tasks]
        conditions = per_task if args.compose else [[name for names in per_task for name in names]]
    weights = args.scale if len(args.scale) == len(conditions) else args.scale[:1] * len(conditions)
    n = args.n_samples
    encoded = conditioner.encode(conditions + [[]], device)  # the conditions, then the null condition
    contexts = [c.expand(n, -1, -1) for c in encoded[:-1].split(1)]
    guided = len(contexts) > 1 or weights[0] != 1.0
    uncond = encoded[-1:].expand(n, -1, -1) if guided else None

    # ---- steering anchor: average of the task anchors, mapped to the sampling space
    anchor = None
    if args.anchors and args.steering > 0:
        anchors = torch.load(args.anchors, map_location="cpu")["anchors"]
        if not args.tasks:
            raise SystemExit("--anchors needs --tasks to know which anchors to use")
        if autoencoders:
            anchor = autoencoders.encode_anchor(anchors, args.tasks)
        else:
            anchor = torch.stack([anchors[t].float() for t in args.tasks]).mean(0, keepdim=True)
        anchor = anchor.to(device) * scale_factor

    generator = torch.Generator(device=device).manual_seed(args.seed)
    samples = DDIMSampler(model).sample(contexts, weights, uncond, steps=args.steps, eta=args.eta,
                                        anchor=anchor, steering_strength=args.steering, generator=generator)
    samples = samples / scale_factor
    if autoencoders:
        prompts = autoencoders.decode(samples, decode_task, prompt_shape)
    else:
        prompts = samples.reshape(n, *prompt_shape)
    prompts = prompts.cpu()

    torch.save({"prompts": prompts, "tasks": args.tasks, "conditions": conditions, "weights": weights,
                "decode_task": decode_task if autoencoders and autoencoders.shared is None else None,
                "steering": args.steering if anchor is not None else 0.0, "seed": args.seed}, args.out)
    print(f"saved {n} prompt(s) of shape {tuple(prompts.shape[1:])} to {args.out}"
          + (f" (steered, lambda={args.steering})" if anchor is not None else ""))
    if args.coop_template:
        write_coop_checkpoint(args.coop_template, prompts[0], args.coop_out or args.out + ".coop.pth.tar")


if __name__ == "__main__":
    main()
