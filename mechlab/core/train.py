"""A small resumable training loop shared by the from-scratch projects.

Data comes from ``batch_fn(step)``, which must be deterministic in ``step``: a run resumed
from a checkpoint then sees exactly the batches it would have seen without interruption.
``batch_fn`` may return NumPy arrays or tensors. If ``tcfg['data_workers'] > 0`` batches are
generated ahead of time in worker processes (``batch_fn`` must then be picklable).
"""
from __future__ import annotations

import math
import os
from typing import Callable, Optional

import numpy as np
import torch

from .prefetch import Prefetcher
from .utils import Timer, append_jsonl, ensure_dir


def _lr_at(step: int, steps: int, lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, steps - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def train(
    model: torch.nn.Module,
    batch_fn: Callable[[int], dict],
    loss_fn: Callable[[torch.nn.Module, dict], tuple],
    tcfg: dict,
    out_dir: str | os.PathLike,
    device,
    eval_fn: Optional[Callable[[torch.nn.Module, int], dict]] = None,
    verbose: bool = True,
) -> dict:
    """Train ``model``; returns the last logged metrics. Writes ckpt.pt, model.pt, train_log.jsonl."""
    out = ensure_dir(out_dir)
    steps = int(tcfg.get("steps", 1000))
    lr = float(tcfg.get("lr", 3e-4))
    warmup = int(tcfg.get("warmup", min(500, steps // 10)))
    wd = float(tcfg.get("weight_decay", 0.01))
    clip = float(tcfg.get("grad_clip", 1.0))
    ckpt_every = int(tcfg.get("ckpt_every", 1000))
    log_every = int(tcfg.get("log_every", 100))
    eval_every = int(tcfg.get("eval_every", 0))
    snapshot_every = int(tcfg.get("snapshot_every", 0))
    workers = int(tcfg.get("data_workers", 0))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"

    final_path = out / "model.pt"
    model.to(device)
    if final_path.exists():
        model.load_state_dict(torch.load(final_path, map_location=device))
        model.eval()
        if verbose:
            print(f"[train] {final_path} exists - skipping training")
        return {"skipped": True}

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start = 0
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if use_amp and "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start = int(ck["step"])
        if verbose:
            print(f"[train] resumed from step {start}")

    timer = Timer()
    last: dict = {}
    model.train()
    with Prefetcher(batch_fn, start, steps, workers=workers, ahead=4 * max(1, workers)) as pf:
        for step in range(start, steps):
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, steps, lr, warmup)
            batch = to_device(pf.get(step), device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, metrics = loss_fn(model, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()

            if (step + 1) % log_every == 0 or step == steps - 1:
                last = {"step": step + 1, "loss": float(loss.detach()), "lr": opt.param_groups[0]["lr"],
                        "elapsed_s": round(timer(), 1), **{k: float(v) for k, v in metrics.items()}}
                append_jsonl(out / "train_log.jsonl", last)
                if verbose:
                    extra = " ".join(f"{k}={float(v):.4f}" for k, v in metrics.items())
                    print(f"[train] step {step + 1}/{steps} loss={last['loss']:.4f} {extra} ({timer():.0f}s)", flush=True)
            if eval_fn is not None and eval_every and (step + 1) % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    ev = eval_fn(model, step + 1)
                model.train()
                append_jsonl(out / "eval_log.jsonl", {"step": step + 1, **ev})
                if verbose:
                    print(f"[eval] step {step + 1}: " + " ".join(
                        f"{k}={v:.4f}" for k, v in ev.items() if isinstance(v, float)), flush=True)
            if snapshot_every and (step + 1) % snapshot_every == 0:
                torch.save(model.state_dict(), out / f"snap_{step + 1:07d}.pt")
            if (step + 1) % ckpt_every == 0 and step + 1 < steps:
                tmp = out / "ckpt.pt.tmp"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "step": step + 1}, tmp)
                os.replace(tmp, ckpt_path)

    torch.save(model.state_dict(), final_path)
    if ckpt_path.exists():
        ckpt_path.unlink()
    model.eval()
    return last
