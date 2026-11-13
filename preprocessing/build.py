"""
Builds the dataset the model trains on, `final-*`, from a `combined-*` dataset.

The pipeline, from the recordings to the model:

    notebooks/03-combine-data       cleans IAM-OnDB and DeepWriting
    notebooks/07 and 08             trace the synthetic samples, and weed them
    synthetic.retime_strokes        re-spaces the synthetic samples, and writes combined-*
    preprocessing.build             this script: writes final-*
    notebooks/04-model-improved     trains on it

This step takes out what is in the recordings but is not handwriting, in this order:

    1. marks         puts back the small marks the synthetic samples lost in tracing
    2. defects       drops the samples whose ink is not what their text says
    3. defects       removes stray marks from the samples that are kept
    4. lifts         merges the point that DeepWriting repeats at every pen lift
    5. dequantise    blurs DeepWriting's pixel grid

Steps 4 and 5 read the recordings as they were made, so the source must be a dataset that has
not been through them. Step 1 reads the synthetic images and their labels in `data/synthetic`. Run from the project root:

    uv run python -m preprocessing.build --seed 42
    uv run python -m preprocessing.build --source combined-2026-11-12T10-04-33

It reads the newest `combined-*` dataset unless `--source` names another, and writes the
dataset to `final-<timestamp>` with `report.json`, which says what was done to which sample
(by its position in the source), and two pictures to check it by: `restored-marks.png` and
`dequantising.png`. `preprocessing.audit` checks the
result.
"""

import argparse
import json
from collections import Counter
from datetime import datetime

import numpy as np

from preprocessing import dataset as datasets
from preprocessing.dataset import Dataset
from preprocessing.defects import BY_HAND, find_mismatches, stray_marks, without_strokes
from preprocessing.dequantise import dequantise, grid_of, plot_comparison
from preprocessing.lifts import lifts_repeat, merge_repeated_lifts
from preprocessing.marks import missing_marks, plot_restored, read_labels, with_marks
from synthetic.paths import CLEANED_DIR


def main(source: str | None, seed: int) -> None:
    source_dir = CLEANED_DIR / source if source else datasets.newest("combined")
    samples, texts, sources = datasets.load(source_dir)
    print(f"read {len(samples)} samples from {source_dir.name}")

    if not any(grid_of(sample) is not None for sample in samples):
        raise SystemExit(
            "no sample is on a pixel grid, so this dataset has been dequantised already: "
            "name the dataset before it with --source"
        )

    # ---------------------------------------- restore marks
    labels = read_labels()
    restored = {}
    for i, source in enumerate(sources):
        if source == "synthetic" and (marks := missing_marks(samples[i], labels[texts[i]])):
            samples[i] = with_marks(samples[i], marks)
            restored[i] = marks
    print(
        f"put back {sum(map(len, restored.values()))} small marks "
        f"in {len(restored)} synthetic samples"
    )

    # ---------------------------------------- drop mismatched samples
    reasons = find_mismatches(Dataset(samples, texts, sources), BY_HAND)
    kept = [i for i in range(len(samples)) if i not in reasons]
    samples = {i: samples[i] for i in kept}  # by position in the source, for the report

    counts = Counter(reason.split(":")[0] for reason in reasons.values())
    print(f"dropped {len(reasons)} samples whose ink is not what their text says:")
    for reason, count in counts.most_common():
        print(f"    {count:>4}  {reason}")

    # ---------------------------------------- remove stray marks
    marks = {i: found for i, sample in samples.items() if (found := stray_marks(sample))}
    for i, found in marks.items():
        samples[i] = without_strokes(samples[i], found)
    print(f"removed a stray mark from {len(marks)} samples")

    # ---------------------------------------- merge repeated lifts
    repeating = [i for i, sample in samples.items() if lifts_repeat(sample)]
    points = sum(map(len, samples.values()))
    for i in repeating:
        samples[i] = merge_repeated_lifts(samples[i])
    merged = points - sum(map(len, samples.values()))
    print(f"merged {merged:,} repeated points into the lift before them, in {len(repeating)} samples")

    # ---------------------------------------- dequantise
    grids = {i: unit for i, sample in samples.items() if (unit := grid_of(sample)) is not None}
    before = [samples[i] for i in grids]
    for i, unit in grids.items():
        samples[i] = dequantise(samples[i], unit, seed + i)
    sizes = [float(np.hypot(*unit)) for unit in grids.values()]
    print(f"dequantised {len(grids)} samples, on grids of {min(sizes):.4f} to {max(sizes):.4f} sample units")

    # ---------------------------------------- save
    out = CLEANED_DIR / f"final-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
    final = Dataset(list(samples.values()), [texts[i] for i in kept], [sources[i] for i in kept])
    datasets.save(final, out)

    report = {
        "source": source_dir.name,
        "seed": seed,
        "samples": {"read": len(texts), "written": len(kept)},
        "marks_restored": {i: len(marks) for i, marks in restored.items()},
        "dropped": [{"position": i, "text": texts[i], "reason": reason} for i, reason in sorted(reasons.items())],
        "stray_marks_removed": sorted(marks),
        "lifts_merged": {"samples": len(repeating), "points": merged},
        "dequantised": {"samples": len(grids), "grid_step": [min(sizes), max(sizes)]},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    plot_restored(
        [(texts[i], samples[i], marks) for i, marks in restored.items() if i in samples],
        out / "restored-marks.png",
    )
    plot_comparison(before, [samples[i] for i in grids], list(grids.values()), out / "dequantising.png")
    print(f"saved {len(kept)} samples, report.json and two pictures to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", help="the folder name of the dataset to read")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.source, args.seed)
