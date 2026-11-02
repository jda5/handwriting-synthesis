"""
Lines cut from a real corpus, kept when they hold a rare character.

Wikipedia is the corpus for now, but `candidates` takes any text, so the
InkBound corpus can be plugged in the same way.
"""

import json
import random
import re
import urllib.parse
import urllib.request
from collections.abc import Iterator
from itertools import chain
from pathlib import Path

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "handwriting-synthesis (https://github.com/jda5/handwriting-synthesis)"
PAGES_PER_REQUEST = 20  # the API's limit for extracts

PARENTHETICAL = re.compile(r"\s*\([^()]*\)")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
DISAMBIGUATION = re.compile(r"may (also )?refer to")

# -------------------------------------------------------------------------------- wikipedia


def fetch_wikipedia(n_pages: int) -> str:
    """
    The plain-text introductions of `n_pages` random articles, one per paragraph.

    Notes:
        https://www.mediawiki.org/wiki/Extension:TextExtracts
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": PAGES_PER_REQUEST,
        "prop": "extracts",
        "explaintext": 1,
        "exintro": 1,
        "exlimit": "max",
    }
    url = f"{WIKIPEDIA_API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    pages = []
    while len(pages) < n_pages:
        with urllib.request.urlopen(request, timeout=30) as response:
            # Transient server errors come back as 200s without a "query"; ask again
            pages_found = json.load(response).get("query", {}).get("pages", {})
        extracts = (page.get("extract", "") for page in pages_found.values())
        pages += [extract for extract in extracts if not DISAMBIGUATION.search(extract)]
    return "\n\n".join(pages[:n_pages])


def wikipedia(cache: Path, n_pages: int) -> str:
    """
    The Wikipedia text, fetched once and kept at `cache`. Delete the file to
    fetch a fresh set of articles.
    """
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(fetch_wikipedia(n_pages), encoding="utf-8")
    return cache.read_text(encoding="utf-8")


# -------------------------------------------------------------------------------- lines


def sentences(text: str) -> list[str]:
    return SENTENCE_END.split(PARENTHETICAL.sub("", text))


def cut(
    sentence: str, rng: random.Random, min_length: int, max_length: int
) -> Iterator[str]:
    """
    Cuts a sentence at word boundaries into lines of random length, so the
    lines spread across the allowed lengths rather than all being one size.
    """
    line, limit = [], rng.randint(min_length, max_length)
    for word in sentence.split():
        if line and len(" ".join(line)) + 1 + len(word) > limit:
            yield " ".join(line)
            line, limit = [], rng.randint(min_length, max_length)
        line.append(word)
    yield " ".join(line)


def candidates(
    text: str,
    rng: random.Random,
    rare: set[str],
    min_length: int,
    max_length: int,
) -> list[str]:
    """
    The lines of `text` that hold at least one rare character.
    """

    def keep(line: str) -> bool:
        return min_length <= len(line) <= max_length and bool(set(line) & rare)

    lines = chain.from_iterable(
        cut(s, rng, min_length, max_length) for s in sentences(text)
    )
    return sorted({line for line in lines if keep(line)})
