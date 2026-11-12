"""
Re-spaces the points of the synthetic strokes the way a real pen samples them.

Notebook 07 spaces the points of every InkSight stroke evenly, but a real pen is sampled in
time, so its points bunch up where it slows: in tight curves and at the start and end of each
stroke. This script learns that spacing from the real lines, puts it into the synthetic ones,
and writes a new `combined-*` dataset with the re-spaced synthetic samples in place of the old.
The shapes do not change, only where the points fall along them, and no step is allowed to
cut across a curve by more than `CHORD_TOLERANCE`.

It reads the newest `combined-*` dataset, which must have a `sources.json` marking the
synthetic samples. Run from the project root:

    uv run python -m synthetic.retime_strokes --seed 42

It needs a few minutes of CPU and no GPU.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import HistGradientBoostingRegressor

from synthetic.paths import CLEANED_DIR

# -------------------------------------------------------------------------------- constants

EVEN_STEP = 0.19  # the spacing notebook 07 gives the synthetic points, in sample units
FINE = 0.02  # spacing of the dense resampling that a stroke's shape is read from
CURVATURE_SCALES = (
    0.05,
    0.15,
    0.4,
)  # smoothing before curvature is measured, in sample units
FIT_SAMPLES = 3000  # real lines the spacing model learns from
CHORD_TOLERANCE = 0.025  # how far a step may stray from the curve it spans, in sample units

SOS = np.array([0, 0, 1, 0, 0], dtype=np.float32)
EOS = np.array([0, 0, 0, 0, 1], dtype=np.float32)


class Noise(NamedTuple):
    sd: float  # of the log step, beyond what the shape explains
    rho: float  # correlation between one step's noise and the next


# -------------------------------------------------------------------------------- strokes


def to_strokes(sample: np.ndarray) -> list[np.ndarray]:
    """
    Splits a sample into its pen-down strokes, each an `(N, 2)` array of points.
    """
    points = sample[:-1, :2].cumsum(axis=0).astype(float)
    return np.split(points, np.flatnonzero(sample[:-1, 3]) + 1)[:-1]


def to_sample(strokes: list[np.ndarray]) -> np.ndarray:
    """
    Turns strokes back into an `(N, 5)` sample of `[dx, dy, pen_down, pen_up, eos]` rows.

    The first point is the start token: it sits at the origin, so its offset is zero.
    """
    rows, previous = [], strokes[0][0]
    for stroke in strokes:
        block = np.zeros((len(stroke), 5), dtype=np.float32)
        block[:, :2] = np.diff(np.vstack([previous, stroke]), axis=0)
        block[:-1, 2] = 1
        block[-1, 3] = 1
        rows.append(block)
        previous = stroke[-1]
    rows.append(EOS)
    return np.vstack(rows).astype(np.float32)


def step_lengths(stroke: np.ndarray) -> np.ndarray:
    return np.hypot(*np.diff(stroke, axis=0).T)


def arc_length(stroke: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(step_lengths(stroke))])


def resample(stroke: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Places points at the given distances along a stroke.
    """
    distance = arc_length(stroke)
    return np.column_stack(
        [np.interp(positions, distance, stroke[:, i]) for i in (0, 1)]
    )


def even_out(stroke: np.ndarray, step: float = EVEN_STEP) -> np.ndarray:
    """
    Spaces a stroke's points evenly, as notebook 07 does, which leaves its shape and no timing.
    """
    length = arc_length(stroke)[-1]
    return resample(stroke, np.linspace(0, length, max(2, round(length / step) + 1)))


def median_step(strokes: list[np.ndarray]) -> float:
    steps = np.concatenate([step_lengths(stroke) for stroke in strokes])
    return float(np.median(steps[steps > 0]))


def drawn(stroke: np.ndarray) -> bool:
    """
    Whether a stroke has a length to re-space. A dot does not, and nor does a tiny jiggle that
    ends where it began: evened out, it keeps only its two ends, which coincide.
    """
    return len(stroke) >= 2 and arc_length(even_out(stroke))[-1] > 0


# -------------------------------------------------------------------------------- spacing model


def features(stroke: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Describes an evenly spaced stroke at every `FINE` along it.

    Args:
        stroke: The stroke, evenly spaced, so that its points carry no timing.

    Returns:
        The distance along the stroke of each description, and the descriptions: the log
        curvature at each scale in `CURVATURE_SCALES`, and the distance from the stroke's start
        and to its end (beyond 5 units, the ends make no difference).
    """
    length = arc_length(stroke)[-1]
    grid = np.append(np.arange(0, length, FINE), length)
    points = resample(stroke, grid)

    columns = []
    for scale in CURVATURE_SCALES:
        smooth = gaussian_filter1d(points, scale / FINE, axis=0, mode="nearest")
        dx, dy = np.gradient(smooth, axis=0).T
        turning = np.gradient(np.unwrap(np.arctan2(dy, dx)))
        columns.append(np.log(np.abs(turning) / FINE + 1e-3))

    columns += [np.minimum(grid, 5.0), np.minimum(length - grid, 5.0)]
    return grid, np.column_stack(columns)


class Steps(NamedTuple):
    features: np.ndarray  # the stroke's shape where each step was taken
    log_step: np.ndarray  # each step's length, relative to its line's median, as a log
    stroke: np.ndarray  # which stroke each step belongs to, to find neighbouring steps
    line: np.ndarray  # which line each step belongs to


def collect_steps(samples: list[np.ndarray]) -> Steps:
    """
    Pairs every pen-down step in `samples` with the shape of its stroke where it was taken.

    The shape is read from the stroke after it is evened out, so the spacing model learns only
    from what an evenly spaced synthetic stroke also shows.
    """
    parts, stroke_id = [], 0
    for line, sample in enumerate(samples):
        strokes = [stroke for stroke in to_strokes(sample) if drawn(stroke)]
        if not strokes:
            continue
        typical = median_step(strokes)

        for stroke in strokes:
            steps = step_lengths(stroke)
            distance = arc_length(stroke)
            midway = (distance[:-1] + distance[1:]) / 2 / distance[-1]  # as a fraction

            grid, described = features(even_out(stroke))
            at = np.clip(
                np.round(midway * grid[-1] / FINE).astype(int), 0, len(grid) - 1
            )

            moved = steps > 0
            parts.append(
                (
                    described[at][moved],
                    np.log(steps[moved] / typical),
                    stroke_id,
                    line,
                    moved.sum(),
                )
            )
            stroke_id += 1

    return Steps(
        np.concatenate([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
        np.concatenate([np.full(part[4], part[2]) for part in parts]),
        np.concatenate([np.full(part[4], part[3]) for part in parts]),
    )


def measure_noise(model: HistGradientBoostingRegressor, steps: Steps) -> Noise:
    """
    Measures how far real steps stray from the model, within a line, and how much one step's
    straying carries over to the next.
    """
    residual = steps.log_step - model.predict(steps.features)
    line_mean = np.bincount(steps.line, residual) / np.bincount(steps.line)
    residual = residual - line_mean[steps.line]

    same_stroke = steps.stroke[1:] == steps.stroke[:-1]
    rho = np.corrcoef(residual[1:][same_stroke], residual[:-1][same_stroke])[0, 1]
    return Noise(float(residual.std()), float(rho))


def r_squared(model: HistGradientBoostingRegressor, steps: Steps) -> float:
    residual = steps.log_step - model.predict(steps.features)
    return float(1 - residual.var() / steps.log_step.var())


# -------------------------------------------------------------------------------- retiming


def chord_gap(curve: np.ndarray, grid: np.ndarray, start: float, end: float) -> float:
    """
    Measures how far the straight step from `start` to `end` strays from the curve it spans.

    Args:
        curve: The stroke's points at every distance in `grid`.
        grid: Distances along the stroke.
        start: The distance along the stroke at which the step begins.
        end: The distance at which it ends.

    Returns:
        The greatest distance from a point of the curve between `start` and `end` to the step.
    """
    between = curve[np.searchsorted(grid, start, "right") : np.searchsorted(grid, end)]
    if not len(between):
        return 0.0

    a, b = (np.array([np.interp(d, grid, curve[:, i]) for i in (0, 1)]) for d in (start, end))
    chord = b - a
    along = np.clip((between - a) @ chord / max(chord @ chord, 1e-12), 0, 1)
    return float(np.hypot(*(between - (a + along[:, None] * chord)).T).max())


def place_points(
    grid: np.ndarray,
    curve: np.ndarray,
    log_step: np.ndarray,
    scale: float,
    noise: Noise,
    seed: int,
) -> np.ndarray:
    """
    Walks along a stroke, taking each step at the length the stroke's shape calls for there.

    Args:
        grid: Distances along the stroke at which `log_step` is known.
        curve: The stroke's points at every distance in `grid`.
        log_step: The predicted log step length at each distance in `grid`.
        scale: The step length that `log_step` is relative to.
        noise: How much to vary the steps beyond what the shape calls for.
        seed: Seeds the variation, so that the same seed walks the same way.

    Returns:
        The distance along the stroke of each point, from 0 to its full length.
    """
    rng = np.random.default_rng(seed)
    length = grid[-1]
    position, positions = 0.0, [0.0]
    wobble = rng.normal(0, noise.sd)

    while True:
        # The step's length is read halfway along it, so a step into a tight curve is short
        guess = scale * np.exp(np.interp(position, grid, log_step) + wobble)
        step = scale * np.exp(np.interp(position + guess / 2, grid, log_step) + wobble)

        # A step that would leave only a sliver of the stroke runs to its end instead
        end = position + step
        if end >= length - step / 4:
            end = length

        # A step never cuts across the curve: one that would is shortened until it follows it
        while chord_gap(curve, grid, position, end) > CHORD_TOLERANCE:
            end = position + 0.7 * (end - position)

        positions.append(end)
        if end == length:
            return np.array(positions)
        position = end

        wobble = (
            noise.rho * wobble + noise.sd * np.sqrt(1 - noise.rho**2) * rng.normal()
        )


def retime(
    sample: np.ndarray, model: HistGradientBoostingRegressor, noise: Noise, seed: int
) -> np.ndarray:
    """
    Re-spaces a sample's points as a real pen would have sampled them, keeping its shape.

    The sample's own step is the pen's unhurried step, mid-stroke, and every stroke slows from
    it in curves and at its ends. So a line of many short strokes gains points, as it does when
    a real pen writes it, and one of long sweeping strokes loses some.
    """
    strokes = to_strokes(sample)
    if not any(drawn(stroke) for stroke in strokes):
        return sample
    scale = median_step(strokes)

    retimed = []
    for i, stroke in enumerate(strokes):
        if not drawn(stroke):
            retimed.append(stroke)
            continue

        # The model reads the evened-out stroke, a touch shorter than the stroke itself, so
        # distances along one are matched to the other by the fraction of the way along
        grid, described = features(even_out(stroke))
        stretch = arc_length(stroke)[-1] / grid[-1]
        curve = resample(stroke, grid * stretch)

        positions = place_points(grid, curve, model.predict(described), scale, noise, seed + i)
        retimed.append(resample(stroke, positions * stretch))

    return to_sample(retimed)


# -------------------------------------------------------------------------------- comparison


def spacing_profile(samples: list[np.ndarray]) -> dict[str, str]:
    """
    Summarises how the points of `samples` are spaced: the differences that can be seen.
    """
    spread, first, last, relative, middle, cuts = [], [], [], [], [], []
    for sample in samples:
        strokes = [stroke for stroke in to_strokes(sample) if drawn(stroke)]
        if not strokes:
            continue
        typical = median_step(strokes)
        steps = np.concatenate([step_lengths(stroke) for stroke in strokes])
        spread.append(steps.std() / steps.mean())
        relative.append(steps[steps > 0] / typical)
        for stroke in strokes:
            steps = step_lengths(stroke)
            if len(steps) >= 5 and (steps > 0).all():
                first.append(steps[0] / typical)
                last.append(steps[-1] / typical)

            # Steps more than half a unit from either end of their stroke, in sample units:
            # a line of short strokes has a low median step, which would flatter these
            distance = arc_length(stroke)
            midway = (distance[:-1] + distance[1:]) / 2
            middle.append(steps[(midway > 0.5) & (distance[-1] - midway > 0.5)])

            # Where two long steps meet, how far their point sits off the line joining its
            # neighbours. On a smooth curve, that is four times the gap between step and curve.
            before, here, after = stroke[:-2], stroke[1:-1], stroke[2:]
            chord, offset = after - before, here - before
            off_line = np.abs(chord[:, 0] * offset[:, 1] - chord[:, 1] * offset[:, 0])
            off_line = off_line / np.maximum(np.hypot(*chord.T), 1e-9)
            cuts.append(off_line[(steps[:-1] > typical) & (steps[1:] > typical)] / 4)

    steps = collect_steps(samples)
    curved = steps.features[:, 1] > np.log(
        0.3
    )  # the power law holds away from straight lines
    exponent = np.polyfit(steps.features[curved, 1], steps.log_step[curved], 1)[0]
    low, high = np.percentile(np.concatenate(relative), [5, 95])

    return {
        "spread of spacing within a line": f"{np.median(spread):.2f}",
        "first step of a stroke": f"{np.median(first):.2f}x",
        "last step of a stroke": f"{np.median(last):.2f}x",
        "middle 90% of steps": f"{low:.2f}x to {high:.2f}x",
        "mid-stroke step: typical, long": "{:.2f}, {:.2f} units".format(
            *np.percentile(np.concatenate(middle), [50, 90])
        ),
        "curve cut by a long step (worst tenth)": f"over {np.percentile(np.concatenate(cuts), 90):.3f}",
        "step length ~ curvature^": f"{exponent:.2f}",
    }


def print_table(columns: dict[str, dict[str, str]]):
    rows = next(iter(columns.values())).keys()
    width = max(map(len, rows)) + 2
    print(" " * width + "".join(f"{name:>22}" for name in columns))
    for row in rows:
        print(
            f"{row:<{width}}"
            + "".join(f"{column[row]:>22}" for column in columns.values())
        )


def plot_comparison(
    real: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    titles: list[str],
    path: Path,
):
    """
    Draws a real line, a synthetic line before and after re-spacing, with every point marked,
    and how their step lengths are spread.
    """
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [1, 1, 1, 1.2]}
    )

    for ax, sample, title in zip(axes, (real, before, after), titles):
        for stroke in to_strokes(sample):
            ax.plot(stroke[:, 0], -stroke[:, 1], "-", color="lightgray", linewidth=1)
            ax.plot(stroke[:, 0], -stroke[:, 1], ".", color="black", markersize=3)
        ax.set_title(title, loc="left")
        ax.set_aspect("equal")
        ax.axis("off")

    bins = np.linspace(-3, 2, 60)
    for sample, title, color in zip(
        (real, before, after), titles, ("black", "lightcoral", "crimson")
    ):
        strokes = [stroke for stroke in to_strokes(sample) if drawn(stroke)]
        steps = np.concatenate([step_lengths(stroke) for stroke in strokes])
        steps = steps[steps > 0] / median_step(strokes)
        axes[3].hist(
            np.log(steps),
            bins,
            histtype="step",
            density=True,
            color=color,
            linewidth=1.5,
            label=title,
        )

    axes[3].set_xlabel("log(step length / the line's median step)")
    axes[3].set_yticks([])
    axes[3].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=110)


# -------------------------------------------------------------------------------- main


def split_of(text: str) -> str:
    """
    The split notebook 04 puts a real sample in, by a hash of its text.
    """
    position = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8]) / 2**64
    return "val" if position < 0.05 else "test" if position < 0.1 else "train"


def main(seed: int) -> None:
    rng = np.random.default_rng(seed)

    # ---------------------------------------- load
    source_dir = max(CLEANED_DIR.glob("combined-*"))
    with np.load(source_dir / "strokes_data.npz") as data:
        samples = [data[f"arr_{i}"] for i in range(len(data.files))]
    texts = json.loads((source_dir / "texts.json").read_text(encoding="utf-8"))
    sources = json.loads((source_dir / "sources.json").read_text(encoding="utf-8"))
    print(f"read {len(samples)} samples from {source_dir.name}")

    synthetic = [i for i, source in enumerate(sources) if source == "synthetic"]
    real = {"train": [], "val": [], "test": []}
    for i, (text, source) in enumerate(zip(texts, sources)):
        if source == "real":
            real[split_of(text)].append(i)

    def pick(split: str, count: int) -> list[np.ndarray]:
        chosen = rng.choice(real[split], min(count, len(real[split])), replace=False)
        return [samples[i] for i in chosen]

    # ---------------------------------------- learn the spacing of real lines
    print(f"learning the spacing of {FIT_SAMPLES} real lines")
    train = collect_steps(pick("train", FIT_SAMPLES))
    model = HistGradientBoostingRegressor(
        max_iter=300, max_leaf_nodes=63, random_state=seed
    )
    model.fit(train.features, train.log_step)
    noise = measure_noise(model, collect_steps(pick("val", 400)))

    # ---------------------------------------- check it on lines it has not seen
    test = pick("test", 300)
    print(
        f"on held-out real lines, the shape explains {r_squared(model, collect_steps(test)):.0%} "
        f"of how their spacing varies (noise: sd {noise.sd:.2f}, step-to-step {noise.rho:.2f})\n"
    )
    evened = [
        to_sample([even_out(s) if drawn(s) else s for s in to_strokes(sample)])
        for sample in test
    ]
    rebuilt = [
        retime(sample, model, noise, seed + k) for k, sample in enumerate(evened)
    ]

    # ---------------------------------------- retime the synthetic lines
    before = [samples[i] for i in synthetic]
    after = [
        retime(sample, model, noise, seed + i) for i, sample in zip(synthetic, before)
    ]

    print_table(
        {
            "real lines": spacing_profile(test),
            "real, evened": spacing_profile(evened),
            "real, re-spaced": spacing_profile(rebuilt),
            "synthetic before": spacing_profile(before),
            "synthetic after": spacing_profile(after),
        }
    )

    # ---------------------------------------- save
    retimed = list(samples)
    for i, sample in zip(synthetic, after):
        retimed[i] = sample
    assert all(
        (s[0] == SOS).all() and (s[-1] == EOS).all() and np.isfinite(s).all()
        for s in retimed
    )

    out = CLEANED_DIR / f"combined-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
    out.mkdir()
    np.savez(out / "strokes_data.npz", *retimed)
    (out / "texts.json").write_text(
        json.dumps(texts, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (out / "sources.json").write_text(
        json.dumps(sources, separators=(",", ":")), encoding="utf-8"
    )

    # A synthetic line with loops and straight runs, beside a real line with the same kind of shapes
    shown = next((k for k, i in enumerate(synthetic) if "P" in texts[i]), 0)
    plot_comparison(
        test[0],
        before[shown],
        after[shown],
        ["real", "synthetic, evenly spaced", "synthetic, re-spaced"],
        out / "retiming.png",
    )

    points_before = sum(map(len, before))
    points_after = sum(map(len, after))
    print(f"\nsynthetic points: {points_before:,} before, {points_after:,} after")
    print(f"saved {len(retimed)} samples, and the comparison retiming.png, to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
