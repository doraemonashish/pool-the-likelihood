"""Tiny helpers for writing Markdown reports and figures."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .utils import ensure_dir


def md_table(headers: Sequence[str], rows: Iterable[Sequence], floatfmt: str = ".3f") -> str:
    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def save_fig(fig, path: str | Path) -> str:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)


def verdict_line(name: str, passed: bool, detail: str) -> str:
    return f"- **{name}: {'PASS' if passed else 'FAIL'}**: {detail}"


def mean_sd(values) -> str:
    """'0.951 ± 0.012 (3)' over non-missing values; '-' if none."""
    import numpy as np

    v = [x for x in values if x is not None]
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.3f} (1)"
    return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f} ({len(v)})"
