"""
Which characters each handwriting font draws faithfully, and whether it joins
its letters.

A character is drawable when the font maps it to a glyph of its own with some
ink, and a person reading the specimen sheets did not reject it. Fonts too scrawly to
read reliably are kept in `fonts/dropped`, which is not searched.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import ImageFont
from scipy.ndimage import label

from synthetic.paths import FONTS_DIR

# Glyphs that exist but read as something else, judged from specimen sheets
REJECTED = {
    "Braden Hill": "&",  # reads as "ę"
    "Caroline Mutiboko": "S",  # reads as "f"
    "Dave Mergens": "!",  # a heart
    "Fh Script": "&",  # reads as "ξ"
    "GH Hughes": "z£",  # "z" reads as "p", "£" as "Ŧ"
    "Gochi Hand": "z",  # reads as "3"
    "Hannahs Messy Handwriting": "Kk",  # read as "h"
    "Herbert Cooper": "J&",  # "J" reads as ")", "&" as "+"
    "John Caplin": "@",  # reads as "a"
    "John Williams": "J",  # reads as "V"
    "Kevin Shirley": "&",  # reads as "ę"
    "Moan Hand": "q$",  # "q" reads as "a", "$" as "#"
    "Print Version": "J&",  # "J" reads as "d"
    "Royston Such CAP": "&",  # reads as "4"
    "Ruth Stafford": "v",  # reads as "u"
    "Tony Flores": "£&",  # "£" has a gap after it
    "V Read": "M",  # reads as "H"
    "Vicky Caulfield": "FJ",  # "F" reads as "f", "J" as "V"
}


@dataclass(frozen=True)
class Font:
    path: Path
    chars: frozenset[str]
    joined: bool

    @property
    def name(self) -> str:
        return self.path.stem

    def draws(self, text: str) -> bool:
        return set(text) <= self.chars


def is_joined(font: ImageFont.FreeTypeFont) -> bool:
    """
    Whether the font joins its letters, which it does if "nnn" is one stroke.
    """
    mask = font.getmask("nnn")
    ink = np.asarray(mask).reshape(mask.size[::-1]) > 127
    return label(ink)[1] == 1


def load_font(path: Path, alphabet: set[str]) -> Font:
    """
    Reads the font at `path` and works out which characters of `alphabet` it
    draws faithfully.

    The character map alone is not enough. Some fonts map a character to an
    empty glyph, and demo fonts draw the same "buy the full version" box for
    every character they withhold. A glyph drawn identically for two
    characters is a faithful drawing of neither.
    """
    cmap = TTFont(path, fontNumber=0).getBestCmap()
    font = ImageFont.truetype(path, 64)
    rejected = set(REJECTED.get(path.stem, ""))

    masks = {char: font.getmask(char) for char in alphabet if ord(char) in cmap}
    shapes = Counter((mask.size, bytes(mask)) for mask in masks.values())

    def faithful(char: str) -> bool:
        mask = masks.get(char)
        return (
            mask is not None
            and mask.getbbox() is not None
            and shapes[mask.size, bytes(mask)] == 1
            and char not in rejected
        )

    chars = frozenset({" "} | {char for char in alphabet if faithful(char)})
    return Font(path, chars, is_joined(font))


def load_fonts(alphabet: set[str]) -> list[Font]:
    paths = sorted(FONTS_DIR.glob("*.[ot]tf"))
    return [load_font(path, alphabet) for path in paths]
