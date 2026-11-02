"""
Writes the texts of the synthetic dataset to `data/synthetic/texts.json`, as a
mapping from source to texts. Run from the project root:

    uv run python -m synthetic.generate_text --seed 42

See `synthetic-data.md` for the strategy.
"""

import argparse
import json
import random
from collections import Counter
from statistics import mean

from synthetic import corpus, emails, postcodes, symbols
from synthetic.paths import SYNTHETIC_DIR
from synthetic.sampling import (
    character_counts,
    load_ground_truth,
    rare_characters,
    rarity_sample,
)

SYNTHETIC_SHARE = 0.1  # of the ground truth's size
SYMBOLS = "$@£%"  # wanted characters that the ground truth lacks
SHARES = {"postcodes": 0.25, "emails": 0.15, "symbols": 0.2, "corpus": 0.4}
# Longer than the ground truth's median of 29, to teach longer sequences
MIN_LINE_LENGTH = 24
WIKIPEDIA_PAGES = 500


def report(
    synthetic: dict[str, list[str]], counts: Counter[str], rare: set[str]
) -> None:
    for source, texts in synthetic.items():
        print(f"{source}: {len(texts)} texts, mean length {mean(map(len, texts)):.0f}")
        print(*(f"  {text}" for text in texts[:5]), sep="\n")

    added = character_counts(text for texts in synthetic.values() for text in texts)
    print("\nrare character: ground truth + synthetic")
    for char in sorted(rare, key=counts.__getitem__):
        print(f"  {char!r}: {counts[char]} + {added[char]}")


def main(seed: int) -> None:
    rng = random.Random(seed)
    texts = load_ground_truth()
    counts = character_counts(texts)
    alphabet = set(counts) | set(SYMBOLS)
    rare = rare_characters(counts, alphabet)
    budget = round(SYNTHETIC_SHARE * len(texts))

    # The model's text window fits the longest line it trains on, so a longer
    # line would lengthen every batch
    max_line_length = max(map(len, texts))

    wikipedia = corpus.wikipedia(SYNTHETIC_DIR / "wikipedia.txt", WIKIPEDIA_PAGES)
    pools = {
        "postcodes": postcodes.candidates(),
        "emails": emails.candidates(),
        "symbols": symbols.candidates(rng, size=10 * budget),
        "corpus": corpus.candidates(
            wikipedia, rng, rare, MIN_LINE_LENGTH, max_line_length
        ),
    }

    # The model can only learn characters in its vocabulary
    synthetic = {
        source: rarity_sample(
            [text for text in pool if set(text) <= alphabet],
            counts,
            round(SHARES[source] * budget),
            rng,
        )
        for source, pool in pools.items()
    }

    path = SYNTHETIC_DIR / "texts.json"
    path.write_text(
        json.dumps(synthetic, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    report(synthetic, counts, rare)
    print(f"\nwrote {sum(map(len, synthetic.values()))} texts to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
