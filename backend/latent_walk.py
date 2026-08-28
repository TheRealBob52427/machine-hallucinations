"""
latent_walk.py — Latent-space "Machine Hallucination" renderer.

Pipeline:
  1. Encode all normalized images into the VAE latent space (deterministic mean).
  2. Plan a nearest-neighbour *tour* through latent space (closed loop) so the
     walk dissolves between semantically-close scenes and loops seamlessly.
  3. Spherically interpolate (slerp) between consecutive latent anchors with
     a dwell/eased travel schedule; inject a whisper of Gaussian noise at the
     apex of the morph — this is what makes the transition "hallucinate"
     instead of cross-fading.
  4. Decode to JPEG frames + write a manifest.json that carries a per-frame
     ENERGY envelope (0 = crystallised image, 1 = full morph). The Three.js
     frontend uses this to synchronise particle turbulence with latent motion.

Can be used standalone:
    python -m backend.latent_walk --steps 48 --fps 30 --dwell 0.18
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .config import get_settings

logger = logging.getLogger("mh.walk")
ProgressFn = Callable[[int, int, str], None]
FRAME_PATTERN = "f_%06d.jpg"

# --------------------------------------------------------------------------- #
# Easing + schedule helpers                                                    #
# --------------------------------------------------------------------------- #
def ease_identity(t: float) -> float:
    return t


def ease_smootherstep(t: float) -> float:
    """6t⁵−15t⁴+10t³ — zero velocity AND acceleration at the ends."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease_in_out_sine(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


EASINGS = {
    "identity": ease_identity,
    "smootherstep": ease_smootherstep,
    "sine": ease_in_out_sine,
}


def travel_schedule(i: int, steps: int, dwell: float, easing) -> Tuple[float, float]:
    """
    Map frame index i (of `steps`) → (t, energy).

    t:      interpolation coefficient in [0,1]. Held at 0 for the first
            `dwell` fraction of steps so the image visibly *crystallises*
            before it dissolves again.
    energy: morph intensity in [0,1] — 0 while crystallised, peaks mid-travel.
            Stored in the manifest; the frontend's turbulence follows it.
    Uses endpoint=False sampling so the last frame of the loop hands off
    perfectly to frame 1 (seamless loop).
    """
    u = i / max(steps, 1)
    if u < dwell:
        return 0.0, 0.0
    tr = min(max((u - dwell) / max(1e-6, 1.0 - dwell), 0.0), 1.0)
    return easing(tr), math.sin(math.pi * tr)


# --------------------------------------------------------------------------- #
# Latent-space mathematics                                                     #
# --------------------------------------------------------------------------- #
def slerp(t: float, z0: torch.Tensor, z1: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical linear interpolation between two latent tensors.

    Direction is interpolated along the great-circle (preserves the VAE's
    latent geometry — images stay *sharp* mid-transition instead of turning
    to grey mush as with naive lerp), while the magnitude is interpolated
    linearly to avoid luminance dips.
    """
    v0, v1 = z0.reshape(-1).double(), z1.reshape(-1).double()
    n0, n1 = v0.norm(), v1.norm()
    if n0 < eps or n1 < eps:
        out = torch.lerp(v0, v1, t)
    else:
        u0, u1 = v0 / n0, v1 / n1
        dot = torch.clamp((u0 * u1).sum(), -1.0 + eps, 1.0 - eps)
        if abs(dot.item()) > 0.9995:                    # nearly parallel → lerp
            out = torch.lerp(v0, v1, t)
        else:
            omega = torch.acos(dot)
            so = torch.sin(omega)
            out = (torch.sin((1.0 - t) * omega) / so) * v0 + (torch.sin(t * omega) / so) * v1
        mag = (1.0 - t) * n0 + t * n1                    # linear magnitude path
        out = out / (out.norm() + eps) * mag
    return out.reshape_as(z0).to(z0.dtype)


def plan_route(latents: torch.Tensor, start: int = 0) -> List[int]:
    """
    Greedy nearest-neighbour tour through latent space (cosine distance),
    closed by returning to the start. Morphing between latent-neighbours is
    what makes transitions look intentional instead of chaotic.
    """
    Z = torch.nn.functional.normalize(latents.flatten(1).float(), dim=1)
    remaining = set(range(Z.shape[0])) - {start}
    route = [start]
    while remaining:
        idx = list(remaining)
        d = 1.0 - (Z[route[-1]].unsqueeze(0) * Z[idx]).sum(1)   # cosine distance
        nxt = idx[int(torch.argmin(d))]
        route.append(nxt)
        remaining.remove(nxt)
    route.append(route[0])                                       # close the loop
    return route


def farthest_point_subset(latents: torch.Tensor, k: int) -> List[int]:
    """Cap the number of keyframes while maximising latent-space coverage."""
    n = latents.shape[0]
    if k <= 0 or n <= k:
        return list(range(n))
    Z = latents.flatten(1).float()
    sel = [0]
    dmin = (Z - Z[0]).norm(dim=1)
    for _ in range(k - 1):
        nxt = int(torch.argmax(dmin))
        sel.append(nxt)
        dmin = torch.minimum(dmin, (Z - Z[nxt]).norm(dim=1))
    return sel


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class WalkConfig:
    steps_per_transition: int = 48      # frames between two anchors
    fps: int = 30                       # playback rate written to the manifest
    dwell: float = 0.18                 # fraction of each transition to *hold*
    easing: str = "smootherstep"        # see EASINGS
    interp: str = "slerp"               # "slerp" | "lerp"
    noise_level: float = 0.035          # mid-morph hallucination noise (0 = off)
    seed: int = 42
    max_keyframes: int = 24             # 0 = use every image
    jpeg_quality: Optional[int] = None  # None → settings.jpeg_quality
    export_video: bool = False          # requires `pip install imageio[ffmpeg]`


# --------------------------------------------------------------------------- #
# The walker                                                                   #
# --------------------------------------------------------------------------- #
class LatentWalker:
    """Encodes images, walks the latent space, renders frame sequences."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.device = self._pick_device()
        self._vae = None

    # -- model ---------------------------------------------------------------
    def _pick_device(self) -> torch.device:
        if self.settings.force_cpu:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def vae(self):
        """Lazy-load the VAE (first call downloads ~335 MB, cached afterwards)."""
        if self._vae is None:
            from diffusers import AutoencoderKL  # deferred import → fast boot

            logger.info("Loading VAE '%s' on %s …",
                        self.settings.vae_model_id, self.device)
            vae = AutoencoderKL.from_pretrained(self.settings.vae_model_id)
            vae = vae.to(self.device).eval()
            vae.enable_slicing()               # lower VRAM peaks during decode
            self._vae = vae
        return self._vae

    @property
    def scaling_factor(self) -> float:
        return float(getattr(self.vae.config, "scaling_factor", 0.18215))

    # -- encode / decode -------------------------------------------------------
    @torch.no_grad()
    def encode(self, paths: Sequence[Path], batch_size: int) -> torch.Tensor:
        """Encode images → latent means, scaled by the VAE scaling factor."""
        latents = []
        for s in range(0, len(paths), batch_size):
            chunk = paths[s:s + batch_size]
            x = torch.stack([
                torch.from_numpy(
                    np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                ).permute(2, 0, 1)
                for p in chunk
            ])                                   # (B,3,H,W) in [0,1]
            x = x.mul(2).sub(1).to(self.device)  # → [-1, 1]
            # .mode() = posterior mean → deterministic, artefact-free anchors
            latents.append(self.vae.encode(x).latent_dist.mode().cpu())
            logger.info("  encoded %d/%d", min(s + batch_size, len(paths)), len(paths))
        return torch.cat(latents) * self.scaling_factor

    @torch.no_grad()
    def decode_batch(self, z: torch.Tensor) -> List[Image.Image]:
        """Decode a (B,4,h,w) latent batch → list of PIL RGB images."""
        img = self.vae.decode(z.to(self.device) / self.scaling_factor).sample
        img = (img * 0.5 + 0.5).clamp(0, 1)
        arr = (img.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        return [Image.fromarray(a) for a in arr]

    # -- render ----------------------------------------------------------------
    def render_walk(
        self,
        paths: Sequence[Path],
        cfg: WalkConfig,
        progress: Optional[ProgressFn] = None,
    ) -> dict:
        """Full pipeline: encode → route → interpolate → decode → manifest."""
        if len(paths) < 2:
            raise RuntimeError("Need at least 2 processed images.")
        if cfg.easing not in EASINGS:
            raise ValueError(f"Unknown easing '{cfg.easing}'. Options: {list(EASINGS)}")
        easing = EASINGS[cfg.easing]
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        quality = cfg.jpeg_quality or self.settings.jpeg_quality

        def report(done: int, total: int, msg: str) -> None:
            (progress or (lambda d, t, m: logger.info("[%3d%%] %s",
                                                      int(100 * d / max(t, 1)), m)))(done, total, msg)

        # -- 1. encode ---------------------------------------------------------
        report(0, 1, f"encoding {len(paths)} keyframes on {self.device}")
        latents = self.encode(paths, self.settings.batch_size_encode)

        # -- 2. optional coverage cap + latent tour ----------------------------
        keep = farthest_point_subset(latents, cfg.max_keyframes)
        latents, paths = latents[keep], [paths[i] for i in keep]
        route = plan_route(latents, start=0)
        report(0, 1, f"route: {' → '.join(paths[i].stem for i in route)}")

        # -- 3. walk + render ---------------------------------------------------
        walk_id = time.strftime("walk_%Y%m%d_%H%M%S")
        out_dir = self.settings.frames_dir / walk_id
        out_dir.mkdir(parents=True, exist_ok=True)

        n_transitions = len(route) - 1
        total = n_transitions * cfg.steps_per_transition
        frame_idx, energy = 0, []

        for seg in range(n_transitions):
            z0, z1 = latents[route[seg]], latents[route[seg + 1]]

            # -- 3a. build this transition's latent batch -----------------------
            zs = []
            for i in range(cfg.steps_per_transition):
                t, e = travel_schedule(i, cfg.steps_per_transition, cfg.dwell, easing)
                if cfg.interp == "slerp":
                    z = slerp(t, z0, z1)
                else:
                    z = torch.lerp(z0.reshape(-1), z1.reshape(-1), t).reshape_as(z0)
                # Hallucination noise: only applied at the apex of the morph,
                # scaled by the latent's own std → organic, never destructive.
                if cfg.noise_level > 0 and e > 0:
                    z = z + cfg.noise_level * e * z.std() * torch.randn_like(z)
                zs.append(z)
                energy.append(round(e, 4))
            zs = torch.stack(zs)

            # -- 3b. decode in batches and stream frames to disk ----------------
            for s in range(0, len(zs), self.settings.batch_size_decode):
                for im in self.decode_batch(zs[s:s + self.settings.batch_size_decode]):
                    frame_idx += 1
                    im.save(out_dir / (FRAME_PATTERN % frame_idx).replace("%06d", f"{frame_idx:06d}"),
                            "JPEG", quality=quality)
                report(frame_idx, total,
                       f"transition {seg + 1}/{n_transitions} (frame {frame_idx}/{total})")

        # -- 4. manifest ---------------------------------------------------------
        manifest = {
            "id": walk_id,
            "frame_count": frame_idx,
            "width": self.settings.image_size,
            "height": self.settings.image_size,
            "fps": cfg.fps,
            "file_pattern": FRAME_PATTERN,
            "base_path": f"/frames/{walk_id}",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "keyframes": [paths[i].name for i in route],
            "energy": energy,                      # frontend turbulence envelope
            "params": asdict(cfg),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest))

        # -- 5. atomically point `latest.json` at the new walk -------------------
        pointer = self.settings.frames_dir / "latest.json.tmp"
        pointer.write_text(json.dumps({"id": walk_id}))
        os.replace(pointer, self.settings.frames_dir / "latest.json")

        self._prune_walks(keep=self.settings.keep_walks, current=walk_id)
        report(total, total, "done")
        return manifest

    def _prune_walks(self, keep: int, current: str) -> None:
        """Delete all but the `keep` most recent old walks (disk hygiene)."""
        import shutil
        walks = sorted(
            (d for d in self.settings.frames_dir.iterdir()
             if d.is_dir() and d.name.startswith("walk_") and d.name != current),
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        for d in walks[keep:]:
            shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    ap = argparse.ArgumentParser(description="Render a latent-space hallucination walk.")
    ap.add_argument("--steps", type=int, default=48, help="frames per transition")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dwell", type=float, default=0.18, help="hold fraction 0–0.6")
    ap.add_argument("--easing", choices=list(EASINGS), default="smootherstep")
    ap.add_argument("--interp", choices=["slerp", "lerp"], default="slerp")
    ap.add_argument("--noise", type=float, default=0.035, help="morph noise 0–0.3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-keyframes", type=int, default=24, help="0 = all images")
    args = ap.parse_args()

    paths = sorted(p for p in settings.processed_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if len(paths) < 2:
        raise SystemExit("Need ≥ 2 processed images. Run: python -m backend.ingest")

    walker = LatentWalker(settings)
    manifest = walker.render_walk(paths, WalkConfig(
        steps_per_transition=args.steps, fps=args.fps, dwell=args.dwell,
        easing=args.easing, interp=args.interp, noise_level=args.noise,
        seed=args.seed, max_keyframes=args.max_keyframes,
    ))
    print(f"\n✓ Rendered {manifest['frame_count']} frames → "
          f"{settings.frames_dir / manifest['id']}")


if __name__ == "__main__":
    main()
