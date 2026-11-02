"""
Email addresses built from short names and domains.
"""

from itertools import product

from synthetic.words import COMPANY_DOMAINS, FIRST, LAST, PERSONAL_DOMAINS

PATTERNS = ["{f}.{l}", "{f[0]}{l}", "{f}{l[0]}", "{f}"]
MAX_REPEATS = 2


def repeats(text: str) -> int:
    """
    Extra occurrences of any letter or digit: 'yahoo.com' -> 2 (three o's).
    """
    chars = [char for char in text if char.isalnum()]
    return len(chars) - len(set(chars))


def candidates() -> list[str]:
    """
    Every address the name lists can make, keeping those with few repeated
    characters.
    """
    domains = PERSONAL_DOMAINS + COMPANY_DOMAINS
    combos = product(FIRST, LAST, PATTERNS, domains)
    addresses = {
        f"{pattern.format(f=first, l=last)}@{domain}"
        for first, last, pattern, domain in combos
    }
    return sorted(address for address in addresses if repeats(address) <= MAX_REPEATS)
