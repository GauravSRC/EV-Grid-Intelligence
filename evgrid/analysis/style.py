"""Shared matplotlib styling (validated categorical palette, recessive axes)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evgrid.config import FIG_DIR  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLUSTER_COLORS = {"DEL": PALETTE[0], "MUM": PALETTE[1], "BLR": PALETTE[2], "HYD": PALETTE[3], "CHE": PALETTE[4]}
POLICY_COLORS = {"Baseline": "#8a8984", "AI": PALETTE[0], "Target": PALETTE[7]}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 120, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2.0, "legend.frameon": False, "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb", "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p
