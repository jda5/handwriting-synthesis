from pathlib import Path

from dotenv import dotenv_values
from PIL import Image as PILImage
from PIL import ImageOps
from typesafe_sdk import Image, Noul, Score, TypeSafeClient

# TypeSafe rejects images with either side < 280px (and < 78,400 total pixels),
# so pad small samples onto a white canvas rather than upscaling the strokes.
MIN_SIDE = 280

sample = PILImage.open(Path("out.png")).convert("RGB")
padded = ImageOps.pad(
    sample,
    (max(sample.width, MIN_SIDE), max(sample.height, MIN_SIDE)),
    color="white",
)

config = dotenv_values(".env")

IMAGE = Image.from_pil(padded)
INTENDED_TEXT = "LE11 2HQ"

with TypeSafeClient(
    api_key=config["TYPESAFE_API_KEY"],
    model="jev-latest",
) as client:
    response = client.system_one(
        state={
            "context": "`sample` is a synthetic handwriting sample, made by rendering "
            "`intended_text` in a handwriting-style font. It will be used as training "
            "data, so every character must be drawn correctly and legibly.",
            "sample": IMAGE,
            "intended_text": INTENDED_TEXT,
        },
        questions={
            "legibility": Score(
                instructions="How easy is it to read each character in `sample`?",
                criteria=[
                    "Every character can be read at a glance from its shape alone.",
                    "Every character can be identified from its own shape, but one or two \
                    take a second look because of stylised strokes, smudges or stray marks.",
                    "At least one character can only be worked out from the surrounding \
                    letters or from `intended_text`, not from its own shape.",
                    "At least one character is unreadable, missing, \
                    or looks like a different character.",
                ],
            ),
            "matches_text": Noul(
                instructions="`sample` reads exactly as `intended_text`: every character "
                "is present, in the same order, with the same capitalisation and punctuation.",
            ),
            "missing_glyph": Noul(
                instructions="`sample` contains an empty box, a question mark, a blank gap "
                "or another placeholder shape where a character of `intended_text` should be.",
            ),
        },
        extra_headers={"TypeSafe-Danger-Zone-Images": "true"},
    )

print(response.model_dump_json(indent=2))
