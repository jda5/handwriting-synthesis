"""
Finds the samples whose ink is not what their text says, and the marks that are not writing.

A sample is dropped when its ink and its text disagree:

- **A word is crossed out.** IAM-OnDB's transcripts write `#` where the writer scribbled a word
  out, so the ink under a `#` is a scribble.
- **The text runs onto a second row.** The model writes one row, left to right.
- **The ink stops before the text does**, so that most of the text was never written.

A sample is kept, without the mark, when a tiny mark sits far from the writing: a tap on the
screen, usually at the right-hand edge once the line was finished.

The limits were set by drawing the samples on either side of them. They catch gross faults
only. A line that is missing its last word or two looks like any other line to them, and the
ones that were found by looking are listed in `BY_HAND`.
"""

import numpy as np
from scipy.spatial import cKDTree

from preprocessing.dataset import Dataset, letter_size, to_sample, to_strokes

# -------------------------------------------------------------------------------- constants

CROSSED_OUT = "#"

BACK = 0.3  # a new row starts with a move back across this much of the line's width...
ROW_DROP = 1.5  # ...after which the ink sits this many letter sizes lower

CUT_SHORT = 0.3  # ink per letter, as a share of what is typical, below which the ink stops early

MARK_SIZE = 0.5  # a stray mark is no bigger than this, in letter sizes...
MARK_POINTS = 8  # ...has no more points than this...
MARK_GAP = 5.0  # ...and is this many letter sizes from any other ink, outside the box around it

# Samples to drop whatever the measures say, by their position in the dataset that
# `retime_strokes` writes, with their text (to check the position still means that sample) and
# the reason.

BY_HAND = {
    6113: ("his", "a crossed-out word comes first"),
    12433: ("enjoying the climate and the", "the ink stops at 'clim'"),
    15224: ("to be selected is that", "the ink stops after 'selected'"),
    16302: ("cheaply as at first appeared. Dr.", "the ink stops after 'as'"),
}

# -------------------------------------------------------------------------------- ink against text


def ink_per_letter(sample: np.ndarray, text: str) -> float:
    """
    Measures the length of a sample's strokes, in letter sizes, for each letter or digit of
    its text. Punctuation takes next to no ink, so it is not counted.
    """
    length = sum(np.hypot(*np.diff(stroke, axis=0).T).sum() for stroke in to_strokes(sample))
    letters = sum(character.isalnum() for character in text)
    return length / max(letter_size(sample), 1e-9) / max(letters, 1)


def on_two_rows(sample: np.ndarray) -> bool:
    """
    Says whether a sample's writing runs onto a second row: the pen moves back across a good
    part of the line, and what it then writes sits a row lower than what came before.
    """
    strokes = to_strokes(sample)
    width = np.ptp(np.concatenate(strokes)[:, 0])
    points = sum(map(len, strokes))

    for k in range(1, len(strokes)):
        if strokes[k][0, 0] - strokes[k - 1][-1, 0] > -BACK * width:
            continue

        before, after = np.concatenate(strokes[:k])[:, 1], np.concatenate(strokes[k:])[:, 1]
        if min(len(before), len(after)) < 0.1 * points:
            continue  # a pen going back to dot an i, not to start a row

        size = max(np.subtract(*np.percentile(part, [75, 25])) for part in (before, after))
        if np.median(after) - np.median(before) > ROW_DROP * size:
            return True

    return False


def find_mismatches(
    dataset: Dataset, by_hand: dict[int, tuple[str, str]] | None = None
) -> dict[int, str]:
    """
    Finds the samples whose ink is not what their text says.

    Args:
        dataset: The dataset to look through.
        by_hand: Samples to add to those the measures find, as `BY_HAND` lists them.

    Returns:
        The reason each such sample is dropped, by its position in `dataset`.

    Raises:
        ValueError: If a position in `by_hand` does not hold the sample it was written for.
    """
    ink = np.array([ink_per_letter(s, t) for s, t in zip(dataset.samples, dataset.texts)])
    real = np.array(dataset.sources) == "real"
    typical = np.median(ink[real] if real.any() else ink)  # synthetic samples alone have no real ones

    reasons = {}
    for i, (sample, text) in enumerate(zip(dataset.samples, dataset.texts)):
        if CROSSED_OUT in text:
            reasons[i] = "a word is crossed out"
        elif on_two_rows(sample):
            reasons[i] = "the text runs onto a second row"
        elif ink[i] < CUT_SHORT * typical:
            reasons[i] = "the ink stops before the text does"

    for i, (text, reason) in (by_hand or {}).items():
        if i >= len(dataset.texts) or dataset.texts[i] != text:
            raise ValueError(f"sample {i} is not {text!r}: BY_HAND was written for another dataset")
        reasons.setdefault(i, f"by hand: {reason}")

    return reasons


# -------------------------------------------------------------------------------- stray marks


def stray_marks(sample: np.ndarray) -> list[int]:
    """
    Finds the strokes of a sample that are tiny marks far from the rest of its ink.

    A full stop set well apart from its sentence is tiny and outside the rest of the ink too,
    which is why the mark must also be `MARK_GAP` letter sizes from anything else.

    Returns:
        The positions of those strokes among the sample's strokes.
    """
    strokes = to_strokes(sample)
    size = letter_size(sample)

    marks = []
    for k, stroke in enumerate(strokes):
        if len(stroke) > MARK_POINTS or np.ptp(stroke, axis=0).max() > MARK_SIZE * size:
            continue
        if len(strokes) == 1:
            continue

        rest = np.concatenate(strokes[:k] + strokes[k + 1 :])
        centre = stroke.mean(axis=0)
        outside = ((centre < rest.min(axis=0)) | (centre > rest.max(axis=0))).any()
        if outside and cKDTree(rest).query(stroke)[0].min() > MARK_GAP * size:
            marks.append(k)

    return marks


def without_strokes(sample: np.ndarray, drop: list[int]) -> np.ndarray:
    """
    Rebuilds a sample without the strokes at the positions in `drop`.
    """
    return to_sample([stroke for k, stroke in enumerate(to_strokes(sample)) if k not in drop])
