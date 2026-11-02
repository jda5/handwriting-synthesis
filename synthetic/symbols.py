"""
Snippets of the symbols that B2B copy needs and the ground truth lacks ($, @,
£ and %), in the company of the prices, dates, times and acronyms that
surround them.
"""

import random
from collections.abc import Callable
from string import ascii_uppercase, digits

from synthetic.words import ACRONYMS, COMPANIES, FIRST, LAST, METRICS, MONTHS, WEEKDAYS

TEMPLATES = [
    "£{price}",
    "${price}",
    "{percent}%",
    "@{handle}",
    "Only £{price} per month",
    "From ${price}/month",
    "£{price} + VAT",
    "Was £{price}, now £{price}",
    "Save ${price} on your first order",
    "Save {percent}% today",
    "Up to {percent}% off",
    "{percent}% of customers agree",
    "Get {percent}% off with code {code}",
    "P.S. {percent}% off until {weekday}",
    "{metric} up {percent}% YoY",
    "{metric} of {percent}% in Q{quarter}",
    "Q{quarter} {year} {metric}: £{price}",
    "{acronym} & {acronym}",
    "Follow us @{handle}",
    "DM @{handle} for details",
    "Tag @{handle} to enter",
    "{day} {month} {year}",
    "{day}/{mo}/{year}",
    "{weekday} {day} {month}",
    "RSVP by {day} {month}",
    "Offer ends {day} {month} {year}",
    "Valid until {day}/{mo}/{year}",
    "{time} - {time}",
    "Join us at {time} on {weekday}",
]

FIELDS: dict[str, Callable[[random.Random], str]] = {
    "price": lambda rng: rng.choice(["{}", "{:,}", "{}.99", "{}.50"]).format(
        rng.randint(1, 9999)
    ),
    "percent": lambda rng: rng.choice(["{}", "{}.5"]).format(rng.randint(1, 99)),
    "code": lambda rng: "".join(
        rng.choices(ascii_uppercase + digits, k=rng.randint(4, 6))
    ),
    "handle": lambda rng: rng.choice(
        [
            rng.choice(COMPANIES),
            rng.choice(COMPANIES).capitalize() + "HQ",
            rng.choice(FIRST).capitalize() + rng.choice(LAST).capitalize(),
        ]
    ),
    "acronym": lambda rng: rng.choice(ACRONYMS),
    "metric": lambda rng: rng.choice(METRICS),
    "quarter": lambda rng: str(rng.randint(1, 4)),
    "day": lambda rng: str(rng.randint(1, 28)),
    "mo": lambda rng: f"{rng.randint(1, 12):02d}",
    "month": lambda rng: rng.choice(MONTHS),
    "year": lambda rng: str(rng.randint(1995, 2030)),
    "weekday": lambda rng: rng.choice(WEEKDAYS),
    "time": lambda rng: rng.choice(
        ["{}am", "{}pm", "{}:30am", "{}:15pm", "{}:00"]
    ).format(rng.randint(1, 12)),
}


class Fields:
    """
    Fills a template's placeholders lazily: `"£{price}".format_map(Fields(rng))`.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    def __getitem__(self, name: str) -> str:
        return FIELDS[name](self.rng)


def candidates(rng: random.Random, size: int) -> list[str]:
    """
    Up to `size` distinct snippets, each a random template filled with random
    values.
    """
    fields = Fields(rng)
    snippets = {rng.choice(TEMPLATES).format_map(fields) for _ in range(size)}
    return sorted(snippets)
