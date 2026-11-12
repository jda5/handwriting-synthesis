"""
Reads and writes the datasets in `data/cleaned`, and takes their samples apart.

A dataset is a folder that holds

    strokes_data.npz    one `(N, 5)` array per sample: rows of `[dx, dy, pen_down, pen_up, eos]`
    texts.json          the text that each sample writes
    sources.json        "real" for a sample recorded from a pen, "synthetic" for one traced by
                        InkSight. Datasets made before the synthetic samples have no such file.
"""

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from synthetic.paths import CLEANED_DIR

SOS = np.array([0, 0, 1, 0, 0], dtype=np.float32)
EOS = np.array([0, 0, 0, 0, 1], dtype=np.float32)


class Dataset(NamedTuple):
    samples: list[np.ndarray]
    texts: list[str]
    sources: list[str]


# -------------------------------------------------------------------------------- files


def newest(prefix: str) -> Path:
    """
    Finds the most recent dataset whose folder name starts with `prefix`.

    The names end in a timestamp, so the most recent one sorts last.
    """
    return max(CLEANED_DIR.glob(f"{prefix}-*"))


def load(path: Path) -> Dataset:
    with np.load(path / "strokes_data.npz") as data:
        samples = [data[f"arr_{i}"] for i in range(len(data.files))]
    texts = json.loads((path / "texts.json").read_text(encoding="utf-8"))

    sources_path = path / "sources.json"
    if sources_path.exists():
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
    else:
        sources = ["real"] * len(texts)

    return Dataset(samples, texts, sources)


def save(dataset: Dataset, path: Path) -> None:
    assert len(dataset.samples) == len(dataset.texts) == len(dataset.sources)
    assert all(
        (s[0] == SOS).all() and (s[-1] == EOS).all() and np.isfinite(s).all()
        for s in dataset.samples
    )

    path.mkdir()
    np.savez(path / "strokes_data.npz", *dataset.samples)
    (path / "texts.json").write_text(
        json.dumps(dataset.texts, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (path / "sources.json").write_text(
        json.dumps(dataset.sources, separators=(",", ":")), encoding="utf-8"
    )


def split_of(text: str, source: str) -> str:
    """
    The split that notebook 04 puts a sample in: by a hash of its text, and synthetic samples
    always train.
    """
    if source == "synthetic":
        return "train"
    position = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8]) / 2**64
    return "val" if position < 0.05 else "test" if position < 0.1 else "train"


# -------------------------------------------------------------------------------- samples


def to_strokes(sample: np.ndarray) -> list[np.ndarray]:
    """
    Splits a sample into its pen-down strokes, each an `(N, 2)` array of points.
    """
    points = sample[:-1, :2].cumsum(axis=0, dtype=float)
    return np.split(points, np.flatnonzero(sample[:-1, 3]) + 1)[:-1]


def to_sample(strokes: list[np.ndarray]) -> np.ndarray:
    """
    Turns strokes back into an `(N, 5)` sample.

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
    return np.vstack([*rows, EOS]).astype(np.float32)


def letter_size(sample: np.ndarray) -> float:
    """
    Measures how big a sample's letters are: the interquartile range of its points' heights.

    Lines are levelled when they are cleaned, so this is close to the handwriting scale of
    notebook 03, and about half the height of a lowercase letter.
    """
    heights = sample[:-1, 1].cumsum(dtype=float)
    return float(np.subtract(*np.percentile(heights, [75, 25])))
