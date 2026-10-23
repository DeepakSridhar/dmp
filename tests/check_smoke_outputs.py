"""Sanity checks on the files written by scripts/smoke_test.sh."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from dmp.checkpoint import load_dmp  # noqa: E402
from dmp.data import ClassNameConditioner, load_repository  # noqa: E402
from dmp.sampler import DDIMSampler  # noqa: E402


def main(out):
    repo = load_repository(os.path.join(out, "data", "repository.pt"))
    anchors = torch.load(os.path.join(out, "anchors.pt"))["anchors"]
    prompt_shape = tuple(repo["prompts"].shape[1:])
    task_means = {t: repo["prompts"][[x == t for x in repo["tasks"]]].mean(0) for t in repo["task_classnames"]}

    samples = {}
    for name in ("task0", "task0_steered", "pair_steered", "pair_composed", "pair_per_task"):
        prompts = torch.load(os.path.join(out, name + ".pt"))["prompts"]
        assert tuple(prompts.shape[1:]) == prompt_shape, (name, prompts.shape)
        assert torch.isfinite(prompts).all(), f"{name}: non-finite values"
        samples[name] = prompts

    def dist(prompts, target):
        return (prompts - target).flatten(1).norm(dim=1).mean().item()

    # steering (Eq. 7) must pull samples toward the anchor
    plain, steered = dist(samples["task0"], anchors["task0"]), dist(samples["task0_steered"], anchors["task0"])
    print(f"distance to the task0 anchor: {plain:.3f} unsteered -> {steered:.3f} steered")
    assert steered < plain, "steering did not move samples toward the anchor"

    # the text condition should matter: task0 samples lie nearest to the task0 prompts
    nearest = {t: dist(samples["task0"], m) for t, m in task_means.items()}
    print("distance of task0 samples to each task's mean prompt:", {t: round(d, 3) for t, d in nearest.items()})
    assert min(nearest, key=nearest.get) == "task0", "samples for task0 are not closest to task0 prompts"

    # sampler invariants on the trained model
    model, ckpt = load_dmp(os.path.join(out, "dmp.pt"))
    cond = ClassNameConditioner.from_file(os.path.join(out, "data", "text_encodings.pt"),
                                          ckpt["config"]["conditioning"]["max_tokens"])
    ctx, null = cond.encode([repo["task_classnames"]["task0"]] * 2), cond.encode([[]] * 2)
    sampler = DDIMSampler(model)

    def run(**kwargs):
        return sampler.sample(ctx, uncond=null, steps=10, generator=torch.Generator().manual_seed(0), **kwargs)

    assert torch.equal(run(weights=[3.0]), run(weights=[3.0])), "sampling is not deterministic for a fixed seed"
    target = torch.zeros(1, *model.sample_shape)
    gaps = [(run(weights=[3.0], anchor=target, steering_strength=s) - target).norm().item() for s in (0.0, 0.2, 0.9)]
    print("distance to a steering target for lambda = 0 / 0.2 / 0.9:", [round(g, 3) for g in gaps])
    assert gaps[0] > gaps[1] > gaps[2], "stronger steering should end closer to the target"
    print("all checks passed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/smoke")
