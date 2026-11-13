"""
Puts back the small marks that the synthetic samples lost when they were traced.

A synthetic sample is InkSight's tracing of an image of its text. The tracing often leaves out
the smallest marks, and notebook 07 then drops some of those it did trace: it keeps as many
small marks in a word as the word has dotted characters, and a comma is not one of them. So
"£3013.99" is written "£3013 99", and the model is taught that a decimal point is a gap.

The image still has every mark. Each sample is laid back over its image, as notebook 08 does,
and a small blob of ink with no stroke on it is drawn in: as a dot if it is round, and as a
short stroke along its length if it is not (a comma, a dash). The new stroke goes into the
sample next to the stroke nearest to it, which is where a hand would have written it.

A dot is two points in the same place, as notebook 07 writes the dots it keeps. Drawn as a
line it has no length and cannot be seen, so draw samples with a marker on every point.
"""

import json
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from preprocessing.dataset import to_sample, to_strokes
from synthetic.paths import IMAGES_DIR, SYNTHETIC_DIR

# -------------------------------------------------------------------------------- constants

STEP = 0.02  # spacing of the points a stroke is measured at, in sample units
RESOLUTION = 12  # pixels per sample unit in a sample's own distance map
CAP = 1.0  # distances are capped here while fitting, in sample units

MARK_SIZE = (0.05, 0.35)  # a mark is a blob of ink between these sizes, in letter heights
UNTRACED = 0.08  # a mark with no stroke this near it was not traced, in letter heights
ROUND = 1.5  # a mark no longer than this many times its width is drawn as a dot


class Fit(NamedTuple):
    scale: float  # image pixels per sample unit
    angle: float  # radians
    shift: np.ndarray  # image pixels

    def to_image(self, points: np.ndarray) -> np.ndarray:
        return self.scale * points @ rotation(self.angle).T + self.shift

    def to_sample(self, points: np.ndarray) -> np.ndarray:
        return (points - self.shift) @ rotation(self.angle) / self.scale


def rotation(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])


# -------------------------------------------------------------------------------- sample over image


def read_labels() -> dict[str, dict]:
    """
    Reads the label of every synthetic image, by its text: no two images have the same text.
    """
    labels = json.loads((SYNTHETIC_DIR / "labels.json").read_text(encoding="utf-8"))
    return {label["text"]: label for label in labels}


def read_ink(label: dict) -> np.ndarray:
    """
    Reads a label's line image as a mask that is true where there is ink.
    """
    return np.array(Image.open(IMAGES_DIR / label["image"]).convert("L")) < 128


def densify(stroke: np.ndarray) -> np.ndarray:
    """
    Re-spaces a stroke to a point every `STEP` along it.
    """
    distance = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(stroke, axis=0).T))])
    if distance[-1] == 0:
        return stroke[:1]
    targets = np.arange(0, distance[-1] + STEP, STEP)
    return np.column_stack([np.interp(targets, distance, stroke[:, i]) for i in (0, 1)])


def read_map(distance_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Looks up a distance map at `x, y` points, giving infinity to points that fall outside it.
    """
    height, width = distance_map.shape
    x, y = np.round(points).astype(int).T
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    values = np.full(len(points), np.inf)
    values[inside] = distance_map[y[inside], x[inside]]
    return values


def fit_to_image(points: np.ndarray, ink: np.ndarray, box: np.ndarray) -> Fit:
    """
    Finds the transform that lays a sample's points over the ink they were traced from.

    This is the fit of notebook 08. A sample is its line's ink after a shift, a rotation and a
    scaling, so one similarity transform puts it back. It starts from the box around the
    line's words, and is nudged until the strokes lie on the ink and the ink under the strokes.

    Args:
        points: The sample's densified points, in sample units.
        ink: The line image's ink mask.
        box: The box around the line's words, as `[left, top, right, bottom]` pixels.

    Returns:
        The `Fit` with the least distance from strokes to ink and from ink to strokes.
    """
    to_ink = ndimage.distance_transform_edt(~ink)  # pixels

    # The same for the strokes, in a map of their own so it is drawn once and not per guess
    low, high = points.min(axis=0), points.max(axis=0)
    origin = low - 2
    cells = np.round((points - origin) * RESOLUTION).astype(int)
    blank = np.ones(np.ceil((high - origin + 2) * RESOLUTION).astype(int)[::-1] + 1, bool)
    blank[cells[:, 1], cells[:, 0]] = False
    to_strokes_map = ndimage.distance_transform_edt(blank) / RESOLUTION  # sample units

    points = points[:: max(1, len(points) // 1500)]
    ink_points = np.argwhere(ink)[:, ::-1].astype(float)
    ink_points = ink_points[:: max(1, len(ink_points) // 1500)]

    def unpack(q: np.ndarray, scale: float, shift: np.ndarray) -> Fit:
        return Fit(scale * np.exp(q[0]), q[1], shift + scale * q[2:])

    def cost(q: np.ndarray, scale: float, shift: np.ndarray) -> float:
        fit = unpack(q, scale, shift)
        strokes_off = read_map(to_ink, fit.to_image(points)) / fit.scale
        ink_off = read_map(to_strokes_map, (fit.to_sample(ink_points) - origin) * RESOLUTION)
        return np.minimum(strokes_off, CAP).mean() + np.minimum(ink_off, CAP).mean()

    # Start with the sample filling the box, sized by its width and then by its height
    best = None
    first_steps = np.vstack([np.zeros(4), np.diag([0.03, 0.02, 0.3, 0.3])])
    for scale in (box[2:] - box[:2]) / np.maximum(high - low, 1e-6):
        shift = (box[:2] + box[2:]) / 2 - scale * (low + high) / 2
        result = minimize(
            cost,
            np.zeros(4),
            (scale, shift),
            method="Nelder-Mead",
            options={"initial_simplex": first_steps, "xatol": 1e-4, "fatol": 1e-5, "maxiter": 600},
        )
        if best is None or result.fun < best[0]:
            best = (result.fun, unpack(result.x, scale, shift))

    return best[1]


# -------------------------------------------------------------------------------- marks


def trace(blob: np.ndarray) -> np.ndarray:
    """
    Draws a blob of ink as a pen would: a dot, or a stroke down the middle of its length.

    Args:
        blob: The `x, y` pixels of the blob.

    Returns:
        The two ends of the stroke, in pixels. They are the same point for a dot. A stroke
        runs downwards, or left to right if it is level, as a comma and a dash are written.
    """
    centre = blob.mean(axis=0)
    if len(blob) < 3:
        return np.array([centre, centre])

    _, _, axes = np.linalg.svd(blob - centre, full_matrices=False)
    along, across = (blob - centre) @ axes[0], (blob - centre) @ axes[1]
    length, width = np.ptp(along) + 1, np.ptp(across) + 1
    if length < ROUND * width:
        return np.array([centre, centre])

    # The pen's tip is as wide as the blob, so its path stops half a width short of each end
    reach = (length - width) / 2 * axes[0]
    ends = np.array([centre - reach, centre + reach])
    downwards = abs(reach[1]) > abs(reach[0])
    return ends[np.argsort(ends[:, 1] if downwards else ends[:, 0])]


def missing_marks(sample: np.ndarray, label: dict) -> list[np.ndarray]:
    """
    Finds the small marks in a sample's image that the sample has no stroke for.

    Args:
        sample: The `(N, 5)` sample.
        label: The label of the image the sample was traced from.

    Returns:
        A stroke for each such mark, in sample units.
    """
    ink = read_ink(label)
    boxes = np.array(label["word_boxes"], float)
    letter_height = np.median(boxes[:, 3] - boxes[:, 1])  # pixels

    strokes = np.concatenate([densify(stroke) for stroke in to_strokes(sample)])
    line_box = np.concatenate([boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)])
    fit = fit_to_image(strokes, ink, line_box)
    traced = cKDTree(fit.to_image(strokes))

    blobs, count = ndimage.label(ink, structure=np.ones((3, 3)))
    marks = []
    for region in ndimage.find_objects(blobs):
        size = max(side.stop - side.start for side in region) / letter_height
        if not MARK_SIZE[0] <= size <= MARK_SIZE[1]:
            continue  # a letter, or a speck left by the renderer

        blob = np.argwhere(blobs[region] > 0)[:, ::-1] + (region[1].start, region[0].start)
        if traced.query(blob)[0].min() > UNTRACED * letter_height:
            marks.append(fit.to_sample(trace(blob.astype(float))))

    return marks


def with_marks(sample: np.ndarray, marks: list[np.ndarray]) -> np.ndarray:
    """
    Adds strokes to a sample, each next to the stroke that is nearest to it: after that
    stroke if the mark is to its right, and before it otherwise.
    """
    strokes = to_strokes(sample)

    for mark in marks:
        centre = mark.mean(axis=0)
        gaps = [np.hypot(*(stroke - centre).T).min() for stroke in strokes]
        nearest = int(np.argmin(gaps))
        after = centre[0] >= strokes[nearest][:, 0].mean()
        strokes.insert(nearest + after, mark)

    return to_sample(strokes)


# -------------------------------------------------------------------------------- comparison


def plot_restored(restored: list[tuple[str, np.ndarray, list[np.ndarray]]], path: Path):
    """
    Draws every sample that had marks put back, with those marks in red.

    Args:
        restored: The text, the finished sample and the marks that were added, for each sample.
        path: Where to save the picture.
    """
    columns = 3
    rows = -(-len(restored) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(21, 2 * rows), squeeze=False)

    for ax, (text, sample, marks) in zip(axes.ravel(), restored):
        for stroke in to_strokes(sample):
            ax.plot(stroke[:, 0], -stroke[:, 1], ".-", color="black", linewidth=0.8, markersize=2)
        for mark in marks:
            ax.plot(mark[:, 0], -mark[:, 1], "o-", color="crimson", markersize=5, linewidth=2)
        ax.set_title(text, fontsize=9, loc="left")
        ax.set_aspect("equal")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
