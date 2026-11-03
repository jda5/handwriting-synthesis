"""
Renders every text in `data/synthetic/texts.json` in a randomly chosen font
that draws all of its characters. The images go to `data/synthetic/images`
and their labels to `data/synthetic/labels.json`, with the box around each
word so the words can be cut out again without OCR. Run from the project root:

    uv run python -m synthetic.render_images --seed 42
"""

import argparse
import json
import random
import shutil
from collections import Counter

from synthetic.fonts import load_fonts
from synthetic.paths import IMAGES_DIR, SYNTHETIC_DIR
from synthetic.render import render_text


def main(seed: int) -> None:
    rng = random.Random(seed)
    synthetic = json.loads((SYNTHETIC_DIR / "texts.json").read_text(encoding="utf-8"))
    alphabet = {char for texts in synthetic.values() for text in texts for char in text}
    fonts = load_fonts(alphabet)

    # Start afresh so that images from an earlier, larger run do not linger
    shutil.rmtree(IMAGES_DIR, ignore_errors=True)
    IMAGES_DIR.mkdir(parents=True)

    labels = []
    for source, texts in synthetic.items():
        for i, text in enumerate(texts):
            capable = [font for font in fonts if font.draws(text)]
            if not capable:
                print(f"skipped {text!r}: no font draws all of it")
                continue

            font = rng.choice(capable)
            image = f"{source}-{i:04d}.png"
            rendering, word_boxes = render_text(text, font, rng)
            rendering.save(IMAGES_DIR / image)
            labels.append(
                {
                    "image": image,
                    "text": text,
                    "source": source,
                    "font": font.name,
                    "word_boxes": word_boxes,
                }
            )

    path = SYNTHETIC_DIR / "labels.json"
    path.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")

    usage = Counter(label["font"] for label in labels)
    print(
        "images per font:",
        *(f"  {name}: {n}" for name, n in usage.most_common()),
        sep="\n",
    )
    print(f"\nrendered {len(labels)} images to {IMAGES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
