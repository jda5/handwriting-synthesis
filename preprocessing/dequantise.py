"""
Blurs the pixel grid out of the lines that were recorded on one.

DeepWriting's lines were recorded on an iPad, so every pen position is a whole number of screen
pixels and every step the pen takes lands on a grid. The model is scored on the density it
gives the true step, and on a grid it can score far better by standing a needle on every grid
point than by learning anything about handwriting. `dequantise` moves each point of such a line
to a random place in the grid cell around it. No point moves further than the recording could
already have been wrong by, so the handwriting looks the same, but its points now fall anywhere
a pen could have been.

IAM-OnDB's lines are on a grid too, of whiteboard units, but one six times finer: too fine for
the narrowest Gaussian that notebook 04's loss allows to pick out. They are left as they are,
and so are the synthetic lines, which are on no grid.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
# -------------------------------------------------------------------------------- constants

FINEST_GRID = 0.02  # a grid with steps shorter than this is left alone, in sample units
TOLERANCE = 0.02  # how far a squared step may be from a whole number of squared grid steps
ON_GRID = 0.99  # the share of a line's steps that must be on the grid

# -------------------------------------------------------------------------------- grid


def grid_of(sample: np.ndarray) -> np.ndarray | None:
    """
    Finds the grid that a sample's points lie on, if they lie on one.

    The pen's shortest step is taken as one step of the grid, or as one step across and one
    down if the pen never took a single step. On a grid every step is a whole number of grid
    steps across and a whole number down, so its squared length is a whole number of squared
    grid steps, whichever way the line was turned when it was cleaned.

    Args:
        sample: The `(N, 5)` sample.

    Returns:
        One step of the grid as a `(dx, dy)` vector, which gives its angle as well as its size,
        or `None` if the sample is on no grid, or on one finer than `FINEST_GRID`.
    """
    steps = sample[1:-1, :2].astype(float)
    steps = steps[steps.any(axis=1)]
    if not len(steps):
        return None

    squared = (steps**2).sum(axis=1)
    x, y = steps[squared.argmin()]

    for unit in np.array([[x, y], [(x - y) / 2, (x + y) / 2]]):
        whole = squared / (unit @ unit)
        on_grid = np.abs(whole - np.round(whole)) < TOLERANCE
        if on_grid.mean() >= ON_GRID and np.hypot(*unit) > FINEST_GRID:
            return unit

    return None


def dequantise(sample: np.ndarray, unit: np.ndarray, seed: int) -> np.ndarray:
    """
    Moves each of a sample's points to a random place in the grid cell around it.

    Args:
        sample: The `(N, 5)` sample, with its points on a grid.
        unit: One step of that grid, from `grid_of`.
        seed: Seeds the noise, so that the same seed moves the points the same way.

    Returns:
        The sample with its offsets recomputed from the moved points. Its pen states, and the
        zero offset of its first row, are unchanged.
    """
    rng = np.random.default_rng(seed)
    points = sample[:-1, :2].cumsum(axis=0, dtype=float)

    # A cell is a square with `unit` along one side: the noise is drawn along the grid's axes
    axes = np.array([unit, [-unit[1], unit[0]]])
    points += rng.uniform(-0.5, 0.5, points.shape) @ axes

    blurred = sample.copy()
    blurred[:-1, :2] = np.diff(points, axis=0, prepend=points[:1])
    return blurred


def grid_steps(sample: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """
    Measures how far each of a sample's steps goes along the grid, in grid steps.
    """
    return sample[1:-1, :2] @ unit / (unit @ unit)


# -------------------------------------------------------------------------------- comparison


def plot_comparison(
    before: list[np.ndarray], after: list[np.ndarray], units: list[np.ndarray], path: Path
):
    """
    Draws the start of a line before and after, with every point marked, and how far the steps
    of every dequantised line go along the grid.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1, 1, 1.2]})

    for ax, sample, title in zip(axes, (before[0], after[0]), ("on the grid", "dequantised")):
        start = sample[:150]
        points = start[:, :2].cumsum(axis=0)
        for stroke in np.split(points, np.flatnonzero(start[:, 3]) + 1):
            ax.plot(stroke[:, 0], -stroke[:, 1], "-", color="lightgray", linewidth=1)
            ax.plot(stroke[:, 0], -stroke[:, 1], ".", color="black", markersize=3)
        ax.set_title(f"the first 150 points of a line, {title}", loc="left")
        ax.set_aspect("equal")
        ax.axis("off")

    bins = np.arange(-4, 8, 0.1) - 0.05
    for samples, title, color in ((before, "on the grid", "lightcoral"), (after, "dequantised", "crimson")):
        steps = np.concatenate([grid_steps(s, unit) for s, unit in zip(samples, units)])
        axes[2].hist(steps, bins, histtype="step", density=True, color=color, linewidth=1.5, label=title)

    axes[2].set_xlabel("how far a step goes along the grid, in grid steps")
    axes[2].set_yticks([])
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=110)
