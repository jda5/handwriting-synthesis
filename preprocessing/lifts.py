"""
Takes out the point that some recordings repeat when the pen lifts.

An iPad reports a lift as an event of its own, at the place the pen already was. So in
DeepWriting every stroke ends with its last point twice, and the row that carries the lift
never moves. IAM-OnDB puts the lift on the stroke's last point, which is what the model should
see everywhere: otherwise "the pen has stopped dead" means "it is lifting" in one source and
nothing in the other.

This has to run before `dequantise`, which would turn the repeated point into a small step.
"""

import numpy as np

REPEATS = 0.9  # the share of a line's lifts that must repeat a point for it to be the device


def lifts_repeat(sample: np.ndarray) -> bool:
    """
    Says whether a sample was recorded by a device that repeats the last point of every stroke.

    A pen that pauses before it lifts repeats a point now and then in any recording, so the
    sample is judged by all of its lifts.
    """
    rows = sample[1:-1]
    lifts = rows[rows[:, 3] == 1]
    return len(lifts) > 0 and (~lifts[:, :2].any(axis=1)).mean() >= REPEATS


def merge_repeated_lifts(sample: np.ndarray) -> np.ndarray:
    """
    Drops each repeated point and lifts the pen at the point before it.

    A dot is a stroke of two points in the same place. It is left alone: without its second
    point it would be a lift with no ink.

    Returns:
        The sample, one row shorter for every stroke of three or more points.
    """
    rows = sample[:-1]
    down = rows[:, 2] == 1

    repeated = (rows[:, 3] == 1) & ~rows[:, :2].any(axis=1)
    repeated[2:] &= down[1:-1] & down[:-2]  # the point before is not the first of its stroke
    repeated[:2] = False
    repeated = np.flatnonzero(repeated)

    merged = sample.copy()
    merged[repeated - 1, 2:4] = (0, 1)
    return np.delete(merged, repeated, axis=0)
