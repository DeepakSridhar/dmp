"""Encode class names with the CLIP text encoder to build the text-encoding file.

Every class name is encoded on its own (one conditioning token per class), plus the
empty string for the unconditional embedding. Requires OpenAI CLIP:
    pip install git+https://github.com/openai/CLIP.git
"""
import argparse
import json
import os

from _common import pick_device
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classnames", required=True,
                   help='JSON: a list of class names, or {"task": ["class name", ...]}')
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="ViT-L/14")
    p.add_argument("--device")
    args = p.parse_args()

    import clip

    with open(args.classnames) as f:
        data = json.load(f)
    names = data if isinstance(data, list) else [n for group in data.values() for n in group]
    names = [""] + sorted(set(names) - {""})
    encodings = torch.load(args.out, map_location="cpu") if os.path.exists(args.out) else {}
    todo = [n for n in names if n not in encodings]

    device = pick_device(args.device)
    model, _ = clip.load(args.model, device=device, jit=False)
    with torch.no_grad():
        for i in range(0, len(todo), 256):
            chunk = todo[i:i + 256]
            features = model.encode_text(clip.tokenize(chunk, truncate=True).to(device)).float().cpu()
            encodings.update(zip(chunk, features))
    torch.save(encodings, args.out)
    print(f"encoded {len(todo)} new names; {args.out} now holds {len(encodings)} entries")


if __name__ == "__main__":
    main()
