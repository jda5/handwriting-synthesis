from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/jacob/github.com/jda5/handwriting-synthesis")
FONTS_DIR = ROOT / "fonts"
SYNTHETIC_DATA_DIR = ROOT / "data" / "synthetic"


def render_text(text: str, font_path: Path, size=64, padding=20) -> Image.Image:
    """
    Notes:
        https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
    """
    font = ImageFont.truetype(font_path, size)

    # Get the bounding box of given text relative to the top-left anchor when rendered in font
    left, top, right, bottom = font.getbbox(text)

    # A 2-tuple, containing (width, height) in pixels.
    size = (int(right - left + 2 * padding), int(bottom - top + 2 * padding))

    # Creates an image with a given mode, size, and background color
    img = Image.new(mode="RGB", size=size, color="white")

    # Draw the text
    ImageDraw.Draw(img).text(
        xy=(padding - left, padding - top),
        text=text,
        font=font,
        fill="black",
    )

    return img


def save_sample(img: Image.Image, text: str, font_name: str):
    output_id = sha256(f"{text}_{font_name}".encode()).hexdigest()
    output_dir = SYNTHETIC_DATA_DIR / output_id
    output_dir.mkdir(parents=True, exist_ok=True)

    img.save(output_dir / f"{output_id}.png")
    with open(output_dir / f"{output_id}.txt", "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    text = "Everything"
    font_name = "Caroline Mutiboko"
    font_path = FONTS_DIR / f"{font_name}.ttf"

    img = render_text(text, font_path)
    # save_sample(img, text, font_name)
    img.save("out2.png")
