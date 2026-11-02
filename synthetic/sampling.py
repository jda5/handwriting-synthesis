"""
Sampling that favours the characters the ground-truth dataset lacks.

Every source of synthetic text offers a pool of candidates; `rarity_sample`
draws from it so that candidates holding rarely seen characters are the most
likely to be chosen.
"""

import heapq
import json
import random
from collections import Counter
from collections.abc import Iterable
from itertools import chain

from synthetic.paths import CLEANED_DIR

# A character is rare if it makes up less than this share of the ground truth
RARE_SHARE = 1e-3


def load_ground_truth() -> list[str]:
    """
    The texts of the latest cleaned dataset.

    The folder names end in a timestamp, so the latest one sorts last.
    """
    path = max(CLEANED_DIR.glob("combined-*")) / "texts.json"
    return json.loads(path.read_text(encoding="utf-8"))


def character_counts(texts: Iterable[str]) -> Counter[str]:
    return Counter(chain.from_iterable(texts))


def rare_characters(counts: Counter[str], alphabet: set[str]) -> set[str]:
    """
    The characters of `alphabet` that are rare in, or absent from, the ground truth.
    """
    threshold = RARE_SHARE * counts.total()
    return {char for char in alphabet if counts[char] < threshold}


def rarity(text: str, counts: Counter[str]) -> float:
    """
    How much `text` would help the dataset.

    Each character scores the inverse of how often it has been seen, so an
    unseen character scores 1 and a common one next to nothing.
    """
    return sum(1 / (counts[char] + 1) for char in text)


def weighted_sample[T](
    items: Iterable[T], weights: Iterable[float], k: int, rng: random.Random
) -> list[T]:
    """
    Samples `k` items without replacement, with probability proportional to weight.

    Each item draws an exponential key with its weight as the rate and the `k`
    smallest keys win. This runs in O(n log k), which matters for the 1.6
    million postcodes.

    Notes:
        Efraimidis & Spirakis, "Weighted random sampling with a reservoir" (2006)
    """
    keyed = ((rng.expovariate(weight), item) for weight, item in zip(weights, items))
    return [item for _, item in heapq.nsmallest(k, keyed)]


def rarity_sample(
    pool: list[str], counts: Counter[str], k: int, rng: random.Random
) -> list[str]:
    return weighted_sample(pool, (rarity(text, counts) for text in pool), k, rng)
