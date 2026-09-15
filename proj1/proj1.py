import csv
import math
from pathlib import Path
import numpy as np
from skimage import io

root = Path(__file__).resolve().parent
input_dir = root / "proj1_data"
output_dir = root / "assets"
search_radius = 15
refine_radius = 3
border = 0.1
smallest_size = 300


def read_channels(path):
    plate = io.imread(path)
    plate = plate.astype(np.float32) / np.iinfo(plate.dtype).max # Normalize pixel values to [0, 1]

    height = plate.shape[0] // 3
    blue = plate[:height]
    green = plate[height:2 * height]
    red = plate[2 * height:3 * height]

    return blue, green, red


def downsample(image):
    height, width = image.shape
    weights = (1, 4, 6, 4, 1) # kernel weights for Gaussian blur

    # Apply horizontal and vertical convolution
    padded = np.pad(image, ((0, 0), (2, 2)), mode="reflect") # Pad image borders with additional values (reflect instead of zeros)
    horizontal = np.zeros_like(image)
    for i, weight in enumerate(weights):
        horizontal += weight * padded[:, i:i + width]
    horizontal /= 16

    padded = np.pad(horizontal, ((2, 2), (0, 0)), mode="reflect")
    vertical = np.zeros_like(image)
    for i, weight in enumerate(weights):
        vertical += weight * padded[i:i + height]
    vertical /= 16

    return np.ascontiguousarray(vertical[::2, ::2]) # Reduce resolution by half after blurring


# Compute edge strength of image
def gradient_magnitude(image):
    padded = np.pad(image, 1, mode="reflect")
    dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2
    dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2
    return np.hypot(dx, dy)


def build_pyramid(image, method, feature):
    levels = [image]
    if method == "pyramid": # Build image pyramid by continually downsampling until we reach the largest allowed size
        while max(levels[-1].shape) > smallest_size:
            levels.append(downsample(levels[-1]))
    if feature == "gradient": # Compute gradient magnitude for each level of the pyramid
        levels = [gradient_magnitude(level) for level in levels]
    return levels


def find_shift(moving, reference, center, radius):
    center_x, center_y = center
    height, width = reference.shape

    # Ignore the black and white borders (could also preprocess images)
    margin_x = math.ceil(width * border)
    margin_y = math.ceil(height * border)
    x0 = margin_x + max(0, center_x + radius)
    x1 = width - margin_x + min(0, center_x - radius)
    y0 = margin_y + max(0, center_y + radius)
    y1 = height - margin_y + min(0, center_y - radius)

    # Flatten reference image into vector and normalize
    target = reference[y0:y1, x0:x1].ravel()
    target = target - target.mean()
    target_norm = float(np.linalg.norm(target))

    # Search for best shift using NCC
    best_shift = center
    best_score = -np.inf
    for dy in range(center_y - radius, center_y + radius + 1):
        for dx in range(center_x - radius, center_x + radius + 1):
            # Flatten moving image into vector and normalize, then compute NCC score
            candidate = moving[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            candidate = candidate.ravel()
            candidate = candidate - candidate.mean()
            denominator = target_norm * float(np.linalg.norm(candidate))
            score = float(np.dot(target, candidate)) / max(denominator, 1e-10)

            if score > best_score:
                best_shift = (dx, dy)
                best_score = score
    return best_shift


# Estimate an initial shift at the smallest level and refine it at each higher level
def align(moving_levels, reference_levels):
    shift = (0, 0)
    last_level = len(reference_levels) - 1
    for level in range(last_level, -1, -1):
        radius = search_radius
        if level < last_level:
            shift = (2 * shift[0], 2 * shift[1])
            radius = refine_radius
        shift = find_shift(moving_levels[level], reference_levels[level], shift, radius)
    return shift


def combine_channels(channels, green_shift, red_shift):
    blue, green, red = channels
    green_x, green_y = green_shift
    red_x, red_y = red_shift
    # Shift green and red images to align with blue
    green = np.roll(green, (green_y, green_x), axis=(0, 1))
    red = np.roll(red, (red_y, red_x), axis=(0, 1))

    # Remove wrapping introduced by np.roll
    height, width = blue.shape
    x0 = max(0, green_x, red_x)
    x1 = width + min(0, green_x, red_x)
    y0 = max(0, green_y, red_y)
    y1 = height + min(0, green_y, red_y)
    return np.stack((red[y0:y1, x0:x1], green[y0:y1, x0:x1], blue[y0:y1, x0:x1]), axis=-1)


def process_image(channels, method, feature):
    blue, green, red = [build_pyramid(channel, method, feature) for channel in channels]
    green_shift = align(green, blue)
    red_shift = align(red, blue)
    rgb = combine_channels(channels, green_shift, red_shift)
    return rgb, green_shift, red_shift


def main():
    files = []
    for path in input_dir.iterdir():
        if path.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff"):
            files.append(path)

    results = []
    for path in files:
        channels = read_channels(path)
        method = "single" if path.suffix.lower() in (".jpg", ".jpeg") else "pyramid"
        features = ("raw", "gradient") if path.stem == "emir" else ("raw",)
        for feature in features:
            rgb, green_shift, red_shift = process_image(channels, method, feature)

            # Save the reconstructed image
            filename = path.stem + ("-gradient" if feature == "gradient" else "") + ".jpg"
            pixels = np.uint8(np.clip(rgb, 0, 1) * 255 + 0.5)
            io.imsave(output_dir / filename, pixels, check_contrast=False)

            results.append([path.stem, method, feature, *green_shift, *red_shift, filename])
            print(f"{path.name} [{method}, {feature}]: "
                  f"G {green_shift}, R {red_shift}")

    with (output_dir / "offsets.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["image", "method", "feature", "green_x", "green_y",
                         "red_x", "red_y", "output"])
        writer.writerows(results)


if __name__ == "__main__":
    main()
