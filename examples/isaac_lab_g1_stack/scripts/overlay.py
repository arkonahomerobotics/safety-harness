"""Caption overlay + mp4 writer for the demo-clip scripts' `--video` flag.

Reconstructed 2026-10-01. This file only ever existed directly on the (now-abandoned) Brev GPU box
that originally ran these scripts -- never committed to the repo, so a fresh clone's own `--video`
path was silently unreproducible (the exact gap already flagged once for this example's own
"Reproduce" section; this is the same class of problem, a different file). Rebuilt from the exact
call signature `gate_policy_g1_stack.py`/`gate_policy_anymal.py`/the Franka demo script already call
it with -- not reverse-engineered from the lost original, so the caption *layout* here is new, but
the function signatures and every value they're called with are unchanged, so no caller needed to
change.

Dependencies: Pillow (PIL) for drawing, imageio (+ imageio-ffmpeg) for mp4 writing if available,
otherwise falls back to piping raw frames straight into a real `ffmpeg` subprocess -- Isaac Sim's own
streaming/rendering stack depends on ffmpeg being present, so this fallback should always have
something to use even on a minimal install.
"""

from __future__ import annotations

import subprocess
import shutil
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_BAR_H = 36
_FONT_SIZE = 18
_PERMIT_RGB = (46, 160, 67)
_BLOCK_RGB = (218, 54, 51)
_BAR_RGB = (18, 18, 18)
_TEXT_RGB = (235, 235, 235)


def _font(size: int = _FONT_SIZE) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(
    rgb: np.ndarray,
    title: str,
    action_label: str,
    verdict: str,
    failing_checks: Sequence[str],
    hazard_text: str,
    footer: str,
) -> np.ndarray:
    """One rendered frame (H, W, 3 uint8) -> the same frame with a top title bar, a bottom status
    bar colored by verdict (green permit / red block, with the failing check names when blocked),
    the injected-hazard line, and a footer -- the exact fields every calling script already passes.
    """
    # Top bar: title + hazard, one row. Bottom: status and footer on their own separate rows --
    # a single shared row overlapped the two whenever both were non-trivial length (found by
    # actually rendering a frame and looking at it, not assumed).
    h, w = rgb.shape[0], rgb.shape[1]
    top_h, bottom_h = _BAR_H, 2 * _BAR_H
    img = Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB")
    canvas = Image.new("RGB", (w, h + top_h + bottom_h), _BAR_RGB)
    canvas.paste(img, (0, top_h))
    draw = ImageDraw.Draw(canvas)
    font = _font()

    draw.text((8, 8), title, fill=_TEXT_RGB, font=font)
    hazard_line = hazard_text if hazard_text else "no hazard"
    draw.text((w - 8, 8), hazard_line, fill=_TEXT_RGB, font=font, anchor="ra")

    verdict_rgb = _PERMIT_RGB if verdict.lower() == "permit" else _BLOCK_RGB
    bottom_top = h + top_h
    draw.rectangle([(0, bottom_top), (w, bottom_top + bottom_h)], fill=verdict_rgb)
    status = f"{action_label}  ->  {verdict.upper()}"
    if failing_checks:
        status += "  (" + ", ".join(failing_checks) + ")"
    draw.text((8, bottom_top + 8), status, fill=(255, 255, 255), font=font)
    if footer:
        draw.text((8, bottom_top + _BAR_H + 8), footer, fill=(255, 255, 255), font=font)

    return np.array(canvas)


def write_mp4(frames: Sequence[np.ndarray], path: str, fps: int) -> None:
    """A real mp4 on disk at `path`, `fps` frames/sec, from a list of (H, W, 3) uint8 frames -- the
    exact call every demo script already makes. Tries imageio first (simpler, handles encoder
    selection itself); falls back to a direct ffmpeg subprocess if imageio or its ffmpeg plugin
    isn't installed, since Isaac Sim's own stack depends on a real ffmpeg binary being present
    regardless.
    """
    if not frames:
        raise ValueError("write_mp4 called with an empty frame list")

    try:
        import imageio.v2 as imageio

        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
            for frame in frames:
                writer.append_data(frame.astype(np.uint8))
        return
    except ImportError:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "write_mp4: neither the Python package `imageio` nor a system `ffmpeg` binary is "
            "available -- install one of the two (`pip install imageio imageio-ffmpeg`, or "
            "`apt-get install ffmpeg`) before using --video."
        )

    h, w = frames[0].shape[0], frames[0].shape[1]
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    err = proc.stderr.read()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited {code} writing {path}:\n{err.decode(errors='replace')[-2000:]}")
