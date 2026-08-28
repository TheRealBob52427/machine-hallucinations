"""
ingest.py — Dataset ingestion & normalization.

Reads every supported image in `data/raw` (including large GeoTIFFs),
center-crops to a square, resizes with Lanczos, converts everything to
8-bit RGB, and writes clean PNGs into `data/processed`.

Usage:
    python -m backend.ingest                       # defaults
    python -m backend.ingest --src data/raw --size 512
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

from .config import get_settings

logger = logging.getLogger("mh.ingest")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}

# Satellite GeoTIFFs routinely exceed PIL's decompression-bomb guard.
# The data is local and trusted, so lift it (explicitly).
Image.MAX_IMAGE_PIXELS = None


def _to_rgb(img: Image.Image) -> Image.Image:
    """Normalize any PIL mode to 8-bit RGB (handles 16-bit rasters)."""
    if img.mode == "RGB":
        return img
    if img.mode in ("I;16", "I;16B", "I"):                       # 16-bit single band
        arr = np.asarray(img).astype(np.float32)
        lo, hi = np.percentile(arr, (2, 98))                     # contrast-stretch
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8), "L").convert("RGB")
    if img.mode == "RGBA":                                        # flatten on black
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def preprocess_file(src: Path, dst_dir: Path, size: int) -> Path:
    """Square-crop + resize a single file. Returns the output path."""
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)                        # honor rotation
        img = _to_rgb(img)
        img = ImageOps.fit(img, (size, size), Image.LANCZOS)      # center-crop
        dst = dst_dir / f"{src.stem}.png"
        img.save(dst, "PNG", optimize=True)
        return dst


def ingest(src_dir: Path, dst_dir: Path, size: int) -> Tuple[int, int]:
    """Batch-ingest a folder. Returns (ok_count, fail_count)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not files:
        raise SystemExit(f"No supported images found in {src_dir} "
                         f"(supported: {sorted(SUPPORTED_EXTS)})")
    ok, fail = 0, 0
    for p in files:
        try:
            out = preprocess_file(p, dst_dir, size)
            logger.info("  ✓ %-40s → %s", p.name, out.name)
            ok += 1
        except Exception as exc:  # never let one corrupt file kill the batch
            logger.warning("  ✗ %-40s   (%s)", p.name, exc)
            fail += 1
    logger.info("Ingest complete: %d ok, %d failed → %s", ok, fail, dst_dir)
    return ok, fail


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    ap = argparse.ArgumentParser(description="Normalize the satellite dataset.")
    ap.add_argument("--src", type=Path, default=settings.raw_dir)
    ap.add_argument("--dst", type=Path, default=settings.processed_dir)
    ap.add_argument("--size", type=int, default=settings.image_size)
    args = ap.parse_args()
    ingest(args.src, args.dst, args.size)


if __name__ == "__main__":
    main()
