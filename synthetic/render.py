"""
Renders text in a handwriting font, one character at a time, distorting each
character a little so that no two renderings of a glyph are identical.
"""

import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont, ImageOps

from synthetic.fonts import Font

# A word's bounding box in pixels: left, top, right, bottom
Box = tuple[int, int, int, int]

# The gap, as a fraction of the font size, left either side of a symbol in a
# joined-up font
PEN_LIFT = 0.1


@dataclass(frozen=True)
class Jitter:
    """
    Standard deviations of the per-character distortions. Lengths are
    fractions of the font size.
    """

    rotation: float = 2.0  # degrees
    scale: float = 0.04
    baseline: float = 0.03
    spacing: float = 0.03


def lifts_pen(char: str) -> bool:
    """
    Whether a writer of joined-up script lifts the pen around `char`: any
    symbol but a space or a quote mark, which sit clear of the joins.
    """
    return not char.isalnum() and char not in " '\""


def union(a: Box, b: Box) -> Box:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def render_text(
    text: str,
    font: Font,
    rng: random.Random,
    jitter: Jitter = Jitter(),
    size: int = 64,
    padding: int = 20,
) -> tuple[Image.Image, list[Box]]:
    """
    Draws `text` in black on a white image cropped to the ink plus `padding`.

    Returns the image and the box around the ink of each of the text's words,
    as split by `text.split()`, so the words can be cut out again without OCR.

    Each glyph is drawn on its own tile, scaled and rotated about its left
    baseline point, then pasted at its usual position, nudged along and across
    the line. The spacing nudges accumulate, so each gap between neighbours
    varies while the line as a whole keeps its length.

    A joined-up font carries its connecting strokes to the edges of each
    letter, where they run through any symbol set between two letters. So the
    pen lifts either side of a symbol, as it would for a person.

    Notes:
        https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
    """
    face = ImageFont.truetype(font.path, size)
    tile = 3 * size  # room for any glyph, swashes included
    origin = (size, 2 * size)  # the left baseline point within a tile

    # Ink is white on black until the end, so that pasting a glyph adds ink
    width = round(1.5 * face.getlength(text)) + 2 * tile
    canvas = Image.new("L", (width, tile), 0)

    drift = 0.0
    boxes: list[Box] = []
    for i, char in enumerate(text):
        lift = PEN_LIFT * size if font.joined and lifts_pen(char) else 0.0
        drift += lift + rng.gauss(0, jitter.spacing * size)
        if char == " ":
            continue

        # ---------------------------------------- distort the glyph
        scaled = face.font_variant(size=round(size * (1 + rng.gauss(0, jitter.scale))))
        glyph = Image.new("L", (tile, tile), 0)
        ImageDraw.Draw(glyph).text(origin, char, font=scaled, fill=255, anchor="ls")
        glyph = glyph.rotate(
            rng.gauss(0, jitter.rotation),
            resample=Image.Resampling.BICUBIC,
            center=origin,
        )

        # ---------------------------------------- place it on the line
        x = face.getlength(text[:i]) + drift
        y = rng.gauss(0, jitter.baseline * size)
        left, top = round(x), round(y)
        canvas.paste(255, (left, top), mask=glyph)
        drift += lift

        # ---------------------------------------- grow the word's box
        x0, y0, x1, y1 = glyph.getbbox()
        box = (left + x0, top + y0, left + x1, top + y1)
        starts_word = i == 0 or text[i - 1] == " "
        if starts_word:
            boxes.append(box)
        else:
            boxes[-1] = union(boxes[-1], box)

    # Crop to the ink, then move the boxes to match
    bbox = canvas.getbbox()
    ink = canvas.crop(bbox)
    dx, dy = padding - bbox[0], padding - bbox[1]
    boxes = [(x0 + dx, y0 + dy, x1 + dx, y1 + dy) for x0, y0, x1, y1 in boxes]
    return ImageOps.invert(ImageOps.expand(ink, border=padding, fill=0)), boxes
