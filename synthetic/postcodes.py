"""
UK postcodes, from https://www.doogal.co.uk/UKPostcodes.
"""

from synthetic.paths import POSTCODES_CSV


def candidates() -> list[str]:
    """
    Every postcode whose characters are all different, so that each sample
    covers as many characters as possible.

    The postcode is the first of the CSV's sixty columns and never contains a
    comma, so splitting the line is far quicker than parsing it.
    """
    with open(POSTCODES_CSV, encoding="utf-8-sig") as f:
        next(f)  # header
        postcodes = (line.split(",", 1)[0] for line in f)
        return [
            postcode for postcode in postcodes if len(set(postcode)) == len(postcode)
        ]
