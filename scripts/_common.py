"""Shared bits for the command-line scripts."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402


def pick_device(name=None):
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def infinite(loader):
    while True:
        for batch in loader:
            yield batch
