"""Small shared helpers: configs, seeding, devices, JSON files."""
from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. Overrides look like ``train.steps=200`` (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_device(prefer: str = "auto"):
    import torch

    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def append_jsonl(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return time.time() - self.t0
