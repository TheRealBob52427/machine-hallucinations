"""
config.py — Central configuration.

Every field can be overridden with an environment variable prefixed `MH_`
(e.g. MH_IMAGE_SIZE=768) or a `.env` file in the project root.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MH_", env_file=".env", extra="ignore")

    # --- Layout -------------------------------------------------------------
    data_dirname: str = "data"
    frontend_dirname: str = "frontend"

    # --- Model --------------------------------------------------------------
    # A standalone KL-autoencoder. Its latent space is smooth enough that
    # spherical interpolation between encodings yields continuous,
    # painterly "hallucinated" morphs rather than cross-fades.
    vae_model_id: str = "stabilityai/sd-vae-ft-mse"
    force_cpu: bool = False        # set MH_FORCE_CPU=1 to ignore CUDA/MPS

    # --- Pre-processing -----------------------------------------------------
    image_size: int = 512          # square training size for the VAE

    # --- Render defaults ----------------------------------------------------
    batch_size_encode: int = 8
    batch_size_decode: int = 16    # lower to 4–8 on ≤6 GB VRAM
    jpeg_quality: int = 90
    keep_walks: int = 2            # how many OLD walks to keep on disk

    # --- Derived paths (properties are not read from env) --------------------
    @property
    def root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        return self.root / self.data_dirname

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def frontend_dir(self) -> Path:
        return self.root / self.frontend_dirname

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.frames_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
