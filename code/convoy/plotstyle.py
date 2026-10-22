"""Shared figure style: one palette and look for every plot in the repo.

Palette (validated for colour-vision deficiency on all pairs, light
background): blue, magenta, green, plus neutral greys. No orange or
yellow. Blue is always the proposed controller; series beyond three are
told apart by line style or marker, never by an extra hue.
"""
import matplotlib.pyplot as plt

BLUE = "#2a78d6"      # proposed
MAGENTA = "#d55181"   # main comparison (PI + feedforward, SAC)
GREEN = "#008300"     # secondary comparison / ablation
GREY = "#8a8985"      # baseline
LIGHT = "#c9c8c3"     # reference path, shading
INK = "#2b2a28"       # text
MUTED = "#52514e"


def apply():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9a9993", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "lines.linewidth": 1.8, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
    })
