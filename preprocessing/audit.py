"""
Checks a dataset for what should not be in it, and prints what it finds.

These are the checks that found what `preprocessing.build` takes out, kept so that the next
dataset can be checked the same way. Run from the project root:

    uv run python -m preprocessing.audit                                   # the newest final-*
    uv run python -m preprocessing.audit combined-2026-11-12T10-04-33

On a dataset that `build` wrote, everything under "defects" should be zero, and so should the
samples on a grid and the samples whose lifts repeat a point. The first defect is checked for
synthetic samples only: they are the ones traced from an image. One artefact is left in on
purpose: most IAM-OnDB strokes have 4k+1 points, because its whiteboard pen reported points
four at a time. A model that counts can tell when a lift is possible. Taking it out would mean
cutting real ink off the ends of strokes.
"""

import argparse
import hashlib
import re
from collections import Counter

import numpy as np

from preprocessing import dataset as datasets
from preprocessing.dataset import Dataset, letter_size, split_of
from preprocessing.defects import find_mismatches, stray_marks
from preprocessing.dequantise import grid_of
from preprocessing.lifts import lifts_repeat
from preprocessing.marks import missing_marks, read_labels
from synthetic.paths import CLEANED_DIR

RARE = 50  # a character seen fewer times than this in training is unlikely to be learnt

# -------------------------------------------------------------------------------- checks


def structure(samples: list[np.ndarray]) -> dict[str, int]:
    return {
        "rows without exactly one pen state": sum(int((s[:, 2:].sum(axis=1) != 1).sum()) for s in samples),
        "samples that do not start with the start row": sum((s[0] != datasets.SOS).any() for s in samples),
        "samples that do not end with one eos row": sum((s[-1] != datasets.EOS).any() or s[:-1, 4].any() for s in samples),
        "samples whose last point is not a lift": sum(s[-2, 3] != 1 for s in samples),
        "samples with a value that is not finite": sum(not np.isfinite(s).all() for s in samples),
    }


def sizes(samples: list[np.ndarray], texts: list[str]) -> dict[str, str]:
    points = np.array([len(s) for s in samples])
    per_character = points / np.array([len(t) for t in texts])
    letter = np.array([letter_size(s) for s in samples])
    return {
        "samples": f"{len(samples):,}",
        "points": f"{points.sum():,}",
        "points per character (median)": f"{np.median(per_character):.1f}",
        "letter size (1st to 99th percentile)": "{:.2f} to {:.2f}".format(*np.percentile(letter, [1, 99])),
    }


def artefacts(samples: list[np.ndarray]) -> dict[str, str]:
    lengths = np.concatenate([np.diff(np.flatnonzero(s[:-1, 3]), prepend=-1) for s in samples])
    still = np.concatenate([~s[1:-1, :2].any(axis=1) for s in samples])
    return {
        "samples on a grid": str(sum(grid_of(s) is not None for s in samples)),
        "samples whose lifts repeat a point": str(sum(lifts_repeat(s) for s in samples)),
        "strokes with 4k+1 points (25% by chance)": f"{np.mean(lengths % 4 == 1):.0%}",
        "rows that do not move": f"{still.mean():.2%}",
    }


def defects(dataset: Dataset) -> dict[str, int]:
    reasons = Counter(find_mismatches(dataset).values())
    labels = read_labels()
    untraced = sum(
        bool(missing_marks(sample, labels[text]))
        for sample, text, source in zip(*dataset)
        if source == "synthetic"
    )
    return {
        "its image has a small mark it does not": untraced,
        "a word is crossed out": reasons["a word is crossed out"],
        "the text runs onto a second row": reasons["the text runs onto a second row"],
        "the ink stops before the text does": reasons["the ink stops before the text does"],
        "samples with a stray mark": sum(bool(stray_marks(s)) for s in dataset.samples),
    }


def leakage(dataset: Dataset) -> dict[str, int]:
    """
    Counts what the validation and test samples share with the training samples.
    """
    splits = [split_of(text, source) for text, source in zip(dataset.texts, dataset.sources)]
    letters = [re.sub(r"[^a-z0-9]", "", text.lower()) for text in dataset.texts]
    digests = [hashlib.sha1(sample.tobytes()).hexdigest() for sample in dataset.samples]

    def held_out_in_train(values: list[str]) -> int:
        train = {value for value, split in zip(values, splits) if split == "train"}
        return sum(split != "train" and value in train for value, split in zip(values, splits))

    return {
        "samples in train, val, test": " ".join(str(splits.count(s)) for s in ("train", "val", "test")),
        "samples that are copies of another": len(digests) - len(set(digests)),
        "held-out samples whose strokes are in training": held_out_in_train(digests),
        "held-out texts in training but for case, spaces and punctuation": held_out_in_train(letters),
    }


def rare_characters(dataset: Dataset) -> dict[str, str]:
    counts = {split: Counter() for split in ("train", "val", "test")}
    for text, source in zip(dataset.texts, dataset.sources):
        counts[split_of(text, source)].update(text)

    rare = sorted((count, character) for character, count in counts["train"].items() if count < RARE)
    unseen = sorted((set(counts["val"]) | set(counts["test"])) - set(counts["train"]))
    return {
        f"seen under {RARE} times in training": "  ".join(f"{c} {n}" for n, c in rare) or "none",
        "held out but never in training": " ".join(unseen) or "none",
    }


# -------------------------------------------------------------------------------- report


def print_table(title: str, columns: dict[str, dict]) -> None:
    rows = next(iter(columns.values())).keys()
    width = max(map(len, rows)) + 2
    print(f"\n{title}")
    if len(columns) > 1:
        print(" " * width + "".join(f"{name:>16}" for name in columns))
    for row in rows:
        values = [str(column[row]) for column in columns.values()]
        print(f"{row:<{width}}" + ("".join(f"{value:>16}" for value in values) if len(values) > 1 else values[0]))


def main(name: str | None) -> None:
    path = CLEANED_DIR / name if name else datasets.newest("final")
    dataset = datasets.load(path)
    print(f"{path.name}: {len(dataset.samples)} samples")

    def only(source: str) -> Dataset:
        keep = [i for i, s in enumerate(dataset.sources) if s == source]
        return Dataset(*([column[i] for i in keep] for column in dataset))

    by_source = {source: only(source) for source in ("real", "synthetic") if source in dataset.sources}

    print_table("sizes", {source: sizes(part.samples, part.texts) for source, part in by_source.items()})
    print_table("structure", {source: structure(part.samples) for source, part in by_source.items()})
    print_table("recording artefacts", {source: artefacts(part.samples) for source, part in by_source.items()})
    print_table("defects", {source: defects(part) for source, part in by_source.items()})
    print_table("splits", {"": leakage(dataset)})
    print_table("characters", {"": rare_characters(dataset)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", nargs="?", help="the folder name of the dataset to check")
    main(parser.parse_args().dataset)
