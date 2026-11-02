from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent

CLEANED_DIR = ROOT / "data" / "cleaned"
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
FONTS_DIR = PACKAGE_DIR / "fonts"
POSTCODES_CSV = PACKAGE_DIR / "postcodes.csv"
IMAGES_DIR = SYNTHETIC_DIR / "images"
