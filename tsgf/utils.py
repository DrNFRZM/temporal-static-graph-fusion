"""Seeding, logging, config and device helpers."""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(out_dir: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(out_dir / "run.log", mode="a", encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config asks for cuda but no GPU is visible; use device: auto or cpu")
    return torch.device(name)


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
