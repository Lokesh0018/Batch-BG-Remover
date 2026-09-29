#!/usr/bin/env python3
"""
Batch Background Remover
========================
Fully automated, free, offline batch image background removal.

Uses rembg + U2Net/ISNet (ONNX) to strip backgrounds from every image
in an input folder and saves transparent PNGs to an output folder.

Usage:
    python main.py                     # process all images once
    python main.py --force             # reprocess even if output exists
    python main.py --watch             # monitor folder for new images
    python main.py --model u2netp      # use the lighter model
    python main.py --recursive         # scan subdirectories
    python main.py --help              # show all options
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pathlib
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from PIL import Image, ImageFilter, ExifTags
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

DEFAULT_INPUT = SCRIPT_DIR / "assets"
DEFAULT_OUTPUT = SCRIPT_DIR / "output"
MODELS_DIR = SCRIPT_DIR / "models"

MANIFEST_NAME = ".processing_manifest.json"
FAILED_LOG_NAME = "failed_images.log"

VALID_MODELS = [
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-dis",
    "birefnet-hrs",
    "birefnet-cod",
    "isnet-general-use",
    "isnet-anime",
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProcessingStats:
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def processed(self) -> int:
        return self.completed + self.failed + self.skipped

    @property
    def elapsed(self) -> float:
        return max(time.time() - self.start_time, 0.001)

    @property
    def speed(self) -> float:
        done = self.completed + self.failed
        return (done / self.elapsed) * 60 if done else 0.0

    @property
    def eta_seconds(self) -> float:
        remaining = self.total - self.processed
        if self.speed > 0:
            return (remaining / self.speed) * 60
        return 0.0

    def eta_display(self) -> str:
        s = int(self.eta_seconds)
        if s <= 0:
            return "---"
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m:02d}m"
        if m > 0:
            return f"{m}m {sec:02d}s"
        return f"{sec}s"


@dataclass
class ImageTask:
    source: pathlib.Path
    output: pathlib.Path
    relative: str  # relative path from input root (for display / manifest key)


# ---------------------------------------------------------------------------
# Manifest -- lightweight JSON that tracks what has been processed
# ---------------------------------------------------------------------------

class Manifest:
    """Tracks processed files so re-runs skip unchanged images."""

    def __init__(self, output_dir: pathlib.Path) -> None:
        self.path = output_dir / MANIFEST_NAME
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
            shutil.move(str(tmp), str(self.path))
        except OSError:
            pass

    def is_done(self, task: ImageTask) -> bool:
        """Return True if the source file was already processed and hasn't changed."""
        key = task.relative
        entry = self._data.get(key)
        if entry is None:
            return False
        if not task.output.exists():
            return False
        try:
            stat = task.source.stat()
            return (
                entry.get("size") == stat.st_size
                and entry.get("mtime") == stat.st_mtime
            )
        except OSError:
            return False

    def mark_done(self, task: ImageTask) -> None:
        try:
            stat = task.source.stat()
            self._data[task.relative] = {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "output": str(task.output),
            }
        except OSError:
            pass

    def remove(self, key: str) -> None:
        self._data.pop(key, None)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_images(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    recursive: bool = False,
) -> list[ImageTask]:
    """Walk the input directory and build a list of ImageTask items."""
    tasks: list[ImageTask] = []

    if recursive:
        iterator: Iterator[pathlib.Path] = input_dir.rglob("*")
    else:
        iterator = input_dir.iterdir()

    for p in sorted(iterator):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        rel = p.relative_to(input_dir)
        out_rel = rel.with_suffix(".png")
        out_path = output_dir / out_rel

        tasks.append(ImageTask(source=p, output=out_path, relative=str(rel)))

    return tasks


# ---------------------------------------------------------------------------
# EXIF orientation correction
# ---------------------------------------------------------------------------

def correct_orientation(img: Image.Image) -> Image.Image:
    """Rotate/flip image according to EXIF orientation tag."""
    try:
        exif = img.getexif()
        if not exif:
            return img

        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break

        if orientation_key is None or orientation_key not in exif:
            return img

        orientation = exif[orientation_key]

        transforms = {
            2: (Image.FLIP_LEFT_RIGHT,),
            3: (Image.ROTATE_180,),
            4: (Image.FLIP_TOP_BOTTOM,),
            5: (Image.TRANSPOSE,),
            6: (Image.ROTATE_270,),
            7: (Image.TRANSVERSE,),
            8: (Image.ROTATE_90,),
        }
        for op in transforms.get(orientation, ()):
            img = img.transpose(op)
    except Exception:
        pass
    return img


# ---------------------------------------------------------------------------
# Post-processing: mask refinement for product photography
# ---------------------------------------------------------------------------

def refine_mask(mask: np.ndarray, strength: str = "medium") -> np.ndarray:
    """
    Refine the alpha mask to improve edge quality.

    Handles the common problem of white/light products on light backgrounds
    where the model produces soft, uncertain edges and gray halos.

    Steps:
      1. Apply a slight Gaussian blur to smooth jagged edges.
      2. Remap the mask with contrast stretching to push uncertain
         mid-range values toward fully opaque or fully transparent.
      3. Clamp to remove residual near-zero or near-255 noise.

    strength:
      "light"  - minimal refinement, keeps soft edges
      "medium" - balanced cleanup, good for most product photos
      "strong" - aggressive cleanup, best for white-on-white
    """
    # Profiles: (blur_radius, low_cutoff, high_cutoff)
    profiles = {
        "light":  (0.5, 20, 235),
        "medium": (0.8, 40, 215),
        "strong": (1.0, 60, 200),
    }
    blur_r, lo, hi = profiles.get(strength, profiles["medium"])

    # Step 1: Mild blur to smooth jagged pixel edges
    mask_img = Image.fromarray(mask, mode="L")
    if blur_r > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_r))
    mask = np.array(mask_img, dtype=np.float32)

    # Step 2: Contrast stretch -- push uncertain values toward 0 or 255
    # Linear remap: values <= lo -> 0, values >= hi -> 255
    mask = np.clip((mask - lo) / max(hi - lo, 1) * 255.0, 0, 255)

    return mask.astype(np.uint8)


def detect_low_contrast_bg(img: Image.Image, sample_fraction: float = 0.05) -> bool:
    """
    Heuristic: detect if the image has a near-uniform light background.

    Samples border pixels (top/bottom/left/right edges). If they are
    mostly bright and similar in color, we flag this as a low-contrast
    product photo that needs stronger mask refinement.
    """
    try:
        rgb = img.convert("RGB")
        arr = np.array(rgb)
        h, w = arr.shape[:2]

        border = max(int(min(h, w) * sample_fraction), 2)

        # Gather border strips
        strips = np.concatenate([
            arr[:border, :, :].reshape(-1, 3),       # top
            arr[-border:, :, :].reshape(-1, 3),       # bottom
            arr[:, :border, :].reshape(-1, 3),        # left
            arr[:, -border:, :].reshape(-1, 3),       # right
        ], axis=0)

        mean_brightness = strips.mean()
        std_brightness = strips.std()

        # Light background: mean > 200, low variance
        return mean_brightness > 200 and std_brightness < 30
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def load_rembg_session(model_name: str):
    """Create a rembg session, loading the model once."""
    from rembg import new_session

    # Set the U2NET_HOME env var so the model is cached in our models/ dir
    os.environ.setdefault("U2NET_HOME", str(MODELS_DIR))

    session = new_session(model_name)
    return session


def process_single_image(
    task: ImageTask,
    session,
    failed_log: pathlib.Path,
    alpha_matting: bool = False,
    post_process: bool = True,
    pp_strength: str = "auto",
) -> bool:
    """Remove background from one image. Returns True on success."""
    from rembg import remove

    tmp_path: pathlib.Path | None = None
    try:
        # Open & correct orientation
        img = Image.open(task.source)
        img = correct_orientation(img)

        # Ensure RGB(A) -- some formats like CMYK need conversion
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGBA")

        # Build kwargs for rembg.remove()
        kwargs: dict = {"session": session}

        # Alpha matting produces much smoother edges for product photos,
        # but requires pymatting and more memory.
        if alpha_matting:
            kwargs["alpha_matting"] = True
            kwargs["alpha_matting_foreground_threshold"] = 270
            kwargs["alpha_matting_background_threshold"] = 20
            kwargs["alpha_matting_erode_size"] = 12

        # Remove background
        result: Image.Image = remove(img, **kwargs)

        # Ensure RGBA for transparency
        if result.mode != "RGBA":
            result = result.convert("RGBA")

        # ----- Post-process the alpha channel for better edges -----
        if post_process:
            r, g, b, a = result.split()
            alpha_arr = np.array(a)

            # Determine strength
            if pp_strength == "auto":
                # Auto-detect: if the source has a light uniform background,
                # use stronger refinement
                if detect_low_contrast_bg(img):
                    strength = "strong"
                else:
                    strength = "medium"
            else:
                strength = pp_strength

            alpha_arr = refine_mask(alpha_arr, strength=strength)
            result = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha_arr, mode="L")))

        # Write to a temp file first, then atomically move
        task.output.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_str = tempfile.mkstemp(suffix=".png", dir=str(task.output.parent))
        tmp_path = pathlib.Path(tmp_str)
        os.close(fd)

        result.save(tmp_path, format="PNG")

        # Atomic rename (same filesystem)
        shutil.move(str(tmp_path), str(task.output))
        tmp_path = None  # mark as moved

        return True

    except Exception as exc:
        _log_failure(failed_log, task, exc)
        return False

    finally:
        # Cleanup temp file if it still exists (failure path)
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        # Release memory
        gc.collect()


def _log_failure(log_path: pathlib.Path, task: ImageTask, exc: Exception) -> None:
    msg = f"[FAIL] {task.relative} -- {type(exc).__name__}: {exc}"
    logging.error(msg)
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    tasks: list[ImageTask],
    session,
    manifest: Manifest,
    failed_log: pathlib.Path,
    force: bool = False,
    alpha_matting: bool = False,
    post_process: bool = True,
    pp_strength: str = "auto",
) -> ProcessingStats:
    """Process a list of image tasks sequentially."""
    stats = ProcessingStats(total=len(tasks))

    if not tasks:
        print("\n  No images to process.\n")
        return stats

    bar_format = (
        "  {l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]"
    )
    pbar = tqdm(
        tasks,
        desc="  Processing",
        unit="img",
        bar_format=bar_format,
        ncols=88,
        file=sys.stdout,
    )

    for task in pbar:
        pbar.set_postfix_str(task.source.name, refresh=False)

        # Skip check
        if not force and manifest.is_done(task):
            stats.skipped += 1
            continue

        # Duplicate-output guard
        if not force and task.output.exists():
            stats.skipped += 1
            manifest.mark_done(task)
            continue

        ok = process_single_image(
            task, session, failed_log,
            alpha_matting=alpha_matting,
            post_process=post_process,
            pp_strength=pp_strength,
        )
        if ok:
            stats.completed += 1
            manifest.mark_done(task)
            # Periodically save manifest (every 25 images)
            if stats.completed % 25 == 0:
                manifest.save()
        else:
            stats.failed += 1

    pbar.close()
    manifest.save()
    return stats


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_loop(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    session,
    manifest: Manifest,
    failed_log: pathlib.Path,
    model_name: str,
    recursive: bool,
    force: bool,
    alpha_matting: bool = False,
    post_process: bool = True,
    pp_strength: str = "auto",
    poll_interval: float = 5.0,
) -> None:
    """Continuously monitor the input folder using watchdog."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
    except ImportError:
        print("\n  [ERROR] watchdog is required for --watch mode.")
        print("  Install it:  pip install watchdog\n")
        sys.exit(1)

    processed_in_watch: set[str] = set()
    pending_queue: list[pathlib.Path] = []

    class _Handler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                self._enqueue(pathlib.Path(event.src_path))

        def on_moved(self, event):
            if not event.is_directory:
                self._enqueue(pathlib.Path(event.dest_path))

        def _enqueue(self, p: pathlib.Path):
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                pending_queue.append(p)

    observer = Observer()
    observer.schedule(_Handler(), str(input_dir), recursive=recursive)
    observer.start()

    print(f"\n  Watching: {input_dir}")
    print("  Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(poll_interval)
            if not pending_queue:
                continue

            # Wait a moment for files to finish copying
            time.sleep(2.0)

            new_files = list(pending_queue)
            pending_queue.clear()

            new_tasks: list[ImageTask] = []
            for p in new_files:
                if not p.is_file():
                    continue
                key = str(p.relative_to(input_dir)) if p.is_relative_to(input_dir) else p.name
                if key in processed_in_watch and not force:
                    continue
                rel = p.relative_to(input_dir) if p.is_relative_to(input_dir) else pathlib.Path(p.name)
                out_rel = rel.with_suffix(".png")
                out_path = output_dir / out_rel
                new_tasks.append(ImageTask(source=p, output=out_path, relative=str(rel)))

            if new_tasks:
                print(f"\n  Detected {len(new_tasks)} new image(s).")
                batch_stats = run_batch(
                    new_tasks, session, manifest, failed_log,
                    force=force,
                    alpha_matting=alpha_matting,
                    post_process=post_process,
                    pp_strength=pp_strength,
                )
                for t in new_tasks:
                    processed_in_watch.add(t.relative)
                _print_summary(batch_stats)

    except KeyboardInterrupt:
        print("\n  Watch mode stopped by user.")
    finally:
        observer.stop()
        observer.join()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

BANNER = """
===================================================
       BATCH  BACKGROUND  REMOVER
==================================================="""


def _detect_device() -> str:
    """Best-effort GPU detection."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return "GPU (CUDA)"
        if "DmlExecutionProvider" in providers:
            return "GPU (DirectML)"
    except Exception:
        pass
    return "CPU"


def print_header(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    model_name: str,
    total: int,
    alpha_matting: bool = False,
    post_process: bool = True,
    pp_strength: str = "auto",
) -> None:
    print(BANNER)
    device = _detect_device()
    print(f"  Input:          {input_dir}")
    print(f"  Output:         {output_dir}")
    print(f"  Model:          {model_name}")
    print(f"  Device:         {device}")
    print(f"  Alpha matting:  {'ON' if alpha_matting else 'OFF'}")
    print(f"  Post-process:   {'ON' if post_process else 'OFF'} (strength: {pp_strength})")
    print(f"  Total images:   {total}")
    print()


def _print_summary(stats: ProcessingStats) -> None:
    elapsed = stats.elapsed
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    time_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

    print()
    print("  -- Summary --------------------------------------")
    print(f"  Completed : {stats.completed}")
    print(f"  Failed    : {stats.failed}")
    print(f"  Skipped   : {stats.skipped}")
    print(f"  Speed     : {stats.speed:.1f} images/min")
    print(f"  Duration  : {time_str}")
    print("  --------------------------------------------------")
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="BatchBGRemover",
        description="Batch background removal using rembg + U2Net/ISNet (offline, free).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                 Process all images
  python main.py --force                         Reprocess everything
  python main.py --watch                         Monitor folder for new images
  python main.py --model isnet-general-use       Best accuracy model
  python main.py --model u2netp                  Fastest model
  python main.py --alpha-matting                 Smoother edges (slower)
  python main.py --post-process strong           Aggressive halo cleanup
  python main.py --no-post-process               Disable mask refinement
  python main.py --recursive                     Include subdirectories
  python main.py --input "D:/imgs"               Custom input folder
  python main.py --output "D:/out"               Custom output folder
""",
    )

    parser.add_argument(
        "--input", "-i",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
        help=f"Input directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="birefnet-general",
        choices=VALID_MODELS,
        help="Background removal model (default: birefnet-general)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Reprocess images even if output already exists",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Scan input subdirectories recursively",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Continuously watch the input folder for new images",
    )
    parser.add_argument(
        "--alpha-matting",
        action="store_true",
        default=False,
        help="Enable alpha matting for smoother edges (slower, more memory)",
    )
    parser.add_argument(
        "--post-process",
        type=str,
        nargs="?",
        const="auto",
        default="auto",
        help="Mask refinement strength: auto, light, medium, strong (default: auto)",
    )
    parser.add_argument(
        "--no-post-process",
        action="store_true",
        default=False,
        help="Disable post-processing mask refinement entirely",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Force UTF-8 output on Windows consoles
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args()

    input_dir: pathlib.Path = args.input.resolve()
    output_dir: pathlib.Path = args.output.resolve()
    model_name: str = args.model
    alpha_matting: bool = args.alpha_matting
    do_post_process: bool = not args.no_post_process
    pp_strength: str = args.post_process if args.post_process else "auto"

    # -- Create directories ------------------------------------------------
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    failed_log = output_dir.parent / FAILED_LOG_NAME

    # -- Configure logging -------------------------------------------------
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # -- Discover images ---------------------------------------------------
    tasks = discover_images(input_dir, output_dir, recursive=args.recursive)

    print_header(
        input_dir, output_dir, model_name, len(tasks),
        alpha_matting=alpha_matting,
        post_process=do_post_process,
        pp_strength=pp_strength,
    )

    if not tasks and not args.watch:
        print("  No supported images found in the input directory.")
        print(f"  Place images in: {input_dir}\n")
        sys.exit(0)

    # -- Load model once ---------------------------------------------------
    print("  Loading model (first run may download the model) ...")
    try:
        session = load_rembg_session(model_name)
    except Exception as exc:
        print(f"\n  [ERROR] Failed to load model '{model_name}': {exc}")
        print("  Make sure you have an internet connection for the first run.\n")
        sys.exit(1)
    print("  Model loaded.\n")

    # -- Manifest ----------------------------------------------------------
    manifest = Manifest(output_dir)

    # -- Process -----------------------------------------------------------
    if tasks:
        stats = run_batch(
            tasks, session, manifest, failed_log,
            force=args.force,
            alpha_matting=alpha_matting,
            post_process=do_post_process,
            pp_strength=pp_strength,
        )
        _print_summary(stats)

        if stats.failed > 0:
            print(f"  See {failed_log} for failure details.\n")

    # -- Watch mode --------------------------------------------------------
    if args.watch:
        watch_loop(
            input_dir=input_dir,
            output_dir=output_dir,
            session=session,
            manifest=manifest,
            failed_log=failed_log,
            model_name=model_name,
            recursive=args.recursive,
            force=args.force,
            alpha_matting=alpha_matting,
            post_process=do_post_process,
            pp_strength=pp_strength,
        )


if __name__ == "__main__":
    main()
