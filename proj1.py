"""CS180 Project 1: from-scratch translation alignment with NumPy and Pillow.

All shifts are (x, y): positive x moves right; positive y moves down.
Run `python proj1.py --help` for the single-scale and pyramid options.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image


def to_float(image):
    """Normalize integer images by their dtype range, not observed maximum."""
    if np.issubdtype(image.dtype, np.integer):
        if image.min() < 0:
            raise ValueError("Expected nonnegative image intensities")
        return image.astype(np.float32) / np.iinfo(image.dtype).max
    result = np.asarray(image, dtype=np.float32)
    if not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise ValueError("Floating-point images must be finite and in [0, 1]")
    return result


def read_channels(path):
    with Image.open(path) as image:
        plate = np.asarray(image)
    if plate.ndim == 3:
        if not np.all(plate[..., :3] == plate[..., :1]):
            raise ValueError("Expected a grayscale three-frame plate")
        plate = plate[..., 0]
    if plate.ndim != 2 or min(plate.shape) < 24:
        raise ValueError("Expected a grayscale plate with three usable frames")
    plate = to_float(plate)
    height = plate.shape[0] // 3
    # Discard at most two trailing rows to obtain equal B, G, R frames.
    return tuple(plate[i * height:(i + 1) * height] for i in range(3))


def downsample(image):
    """Binomial blur [1,4,6,4,1]/16 on each axis, then decimate by two."""
    kernel = (1, 4, 6, 4, 1)
    h, w = image.shape
    padded = np.pad(image, ((0, 0), (2, 2)), mode="reflect")
    horizontal = sum(v * padded[:, i:i + w] for i, v in enumerate(kernel)) / 16
    padded = np.pad(horizontal, ((2, 2), (0, 0)), mode="reflect")
    blurred = sum(v * padded[i:i + h] for i, v in enumerate(kernel)) / 16
    # Sampling 0,2,4,... preserves the coordinate origin for odd dimensions.
    return np.ascontiguousarray(blurred[::2, ::2], dtype=np.float32)


def make_pyramid(image, coarsest_size=350):
    levels = [image]
    while max(levels[-1].shape) > coarsest_size and min(levels[-1].shape) > 64:
        levels.append(downsample(levels[-1]))
    return levels


def features(image, kind):
    if kind == "raw":
        return image
    if kind != "gradient":
        raise ValueError("Features must be raw or gradient")
    padded = np.pad(image, 1, mode="reflect")
    gx = (padded[1:-1, 2:] - padded[1:-1, :-2]) * 0.5
    gy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * 0.5
    return np.hypot(gx, gy)


def search_translation(moving, reference, center=(0, 0), radius=15,
                       metric="ncc", crop=0.12, max_samples=0):
    """Nested-loop exhaustive search on fixed, valid interior support.

    All candidates use identical reference coordinates. The support excludes
    borders in BOTH images for EVERY shift, with no circular wraparound.
    Large images use a regular scoring grid, still testing one-pixel shifts.
    max_samples=0 evaluates every interior pixel.
    """
    if moving.shape != reference.shape or moving.ndim != 2:
        raise ValueError("Images must be equally sized 2D arrays")
    if radius < 0 or not 0 <= crop < 0.4 or max_samples < 0:
        raise ValueError("Invalid radius, crop fraction, or sample count")
    if metric not in ("ncc", "l2"):
        raise ValueError("Metric must be ncc or l2")
    h, w = reference.shape
    cx, cy = center
    mx, my = math.ceil(w * crop), math.ceil(h * crop)
    x0, x1 = mx + max(0, cx + radius), w - mx + min(0, cx - radius)
    y0, y1 = my + max(0, cy + radius), h - my + min(0, cy - radius)
    if min(x1 - x0, y1 - y0) < 8:
        raise ValueError("Search window leaves insufficient interior overlap")
    stride = max(1, math.ceil(max(y1 - y0, x1 - x0) / max_samples)) if max_samples else 1
    target = np.ascontiguousarray(reference[y0:y1:stride, x0:x1:stride]).ravel()
    if metric == "ncc":
        target = target - target.mean()
        target_norm = float(np.linalg.norm(target))
        if target_norm < 1e-10:
            raise ValueError("Reference has no contrast in the scoring region")
    best_shift, best_score, best_distance = center, -np.inf, np.inf
    for dy in range(cy - radius, cy + radius + 1):
        for dx in range(cx - radius, cx + radius + 1):
            candidate = np.ascontiguousarray(moving[y0-dy:y1-dy:stride, x0-dx:x1-dx:stride]).ravel()
            if metric == "ncc":
                candidate = candidate - candidate.mean()
                denominator = target_norm * float(np.linalg.norm(candidate))
                score = float(np.dot(target, candidate)) / denominator if denominator > 1e-10 else -1.0
            else:
                difference = target - candidate
                # MSE has the same argmin as L2 for this fixed-size support.
                score = -float(np.dot(difference, difference)) / target.size
            distance = (dx-cx)**2 + (dy-cy)**2
            if score > best_score or (score == best_score and distance < best_distance):
                best_shift, best_score, best_distance = (dx, dy), score, distance
    return best_shift, best_score


def align(moving, reference, method="pyramid", metric="ncc", feature="raw",
          radius=15, refine_radius=3, crop=0.12, coarsest_size=350, max_samples=0):
    if method not in ("single", "pyramid") or coarsest_size < 64:
        raise ValueError("Invalid method or pyramid size")
    moving_levels = make_pyramid(moving, coarsest_size) if method == "pyramid" else [moving]
    reference_levels = make_pyramid(reference, coarsest_size) if method == "pyramid" else [reference]
    shift, trace = (0, 0), []
    for level in reversed(range(len(moving_levels))):
        first = level == len(moving_levels)-1
        if not first:
            shift = (2*shift[0], 2*shift[1])
        shift, score = search_translation(
            features(moving_levels[level], feature), features(reference_levels[level], feature),
            shift, radius if first else refine_radius, metric, crop,
            0 if method == "single" else max_samples)
        trace.append({"level": level, "shape": list(reference_levels[level].shape),
                      "shift_xy": list(shift), "score": score})
    return shift, trace


def compose(channels, green_xy, red_xy):
    """Roll G/R; retain only their valid intersection with B (no wrapped pixels)."""
    blue, green, red = channels
    h, w = blue.shape
    shifts = ((0, 0), green_xy, red_xy)
    x0, x1 = max(s[0] for s in shifts), w + min(s[0] for s in shifts)
    y0, y1 = max(s[1] for s in shifts), h + min(s[1] for s in shifts)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Translations leave no valid overlap")
    green = np.roll(green, (green_xy[1], green_xy[0]), axis=(0, 1))
    red = np.roll(red, (red_xy[1], red_xy[0]), axis=(0, 1))
    rgb = np.stack((red[y0:y1, x0:x1], green[y0:y1, x0:x1], blue[y0:y1, x0:x1]), axis=-1)
    return rgb, (x0, y0, x1, y1)


def save_jpeg(rgb, path, max_dimension=0, quality=92):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1)*255 + 0.5))
    if max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    image.save(path, quality=quality, subsampling=0, optimize=True)


def process(path, output_dir, **options):
    start = perf_counter()
    channels = read_channels(path)
    read_seconds = perf_counter() - start
    green, green_trace = align(channels[1], channels[0], **options)
    red, red_trace = align(channels[2], channels[0], **options)
    align_seconds = perf_counter() - start - read_seconds
    rgb, overlap = compose(channels, green, red)
    output = Path(output_dir) / f"{Path(path).stem}.jpg"
    save_jpeg(rgb, output)
    record = {"name": Path(path).stem, "input": str(path), "output": str(output),
              "channel_shape": list(channels[0].shape), "green_xy": list(green),
              "red_xy": list(red), "overlap_xyxy": list(overlap),
              "alignment_seconds": round(align_seconds, 4),
              "total_seconds": round(perf_counter()-start, 4), "parameters": options,
              "green_trace": green_trace, "red_trace": red_trace}
    print(f"{Path(path).name}: G (x,y)={green}, R (x,y)={red}; "
          f"alignment {align_seconds:.2f}s; total {record['total_seconds']:.2f}s", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Three-frame grayscale JPG/TIFF files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pyramid"))
    parser.add_argument("--method", choices=("single", "pyramid"), default="pyramid")
    parser.add_argument("--metric", choices=("ncc", "l2"), default="ncc")
    parser.add_argument("--feature", choices=("raw", "gradient"), default="raw")
    parser.add_argument("--radius", type=int, default=15)
    parser.add_argument("--refine-radius", type=int, default=3)
    parser.add_argument("--crop", type=float, default=0.12)
    parser.add_argument("--coarsest-size", type=int, default=350)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Maximum scoring-grid side; 0 uses every pixel. Single-scale always uses all pixels.")
    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("inputs", "output_dir")}
    records = [process(path, args.output_dir, **options) for path in sorted(args.inputs)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
