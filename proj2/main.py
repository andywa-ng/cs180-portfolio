import csv
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.signal import convolve2d

from part1 import (Dx, Dy, convolve_four_loops, convolve_two_loops,
                   dog_gradients, gaussian_blur, gaussian_kernel, gradients)
from part2 import (align_to_landmarks, fit_image, hybrid_image,
                   log_spectrum, multiresolution_blend, unsharp_mask)

root = Path(__file__).resolve().parent
input_dir = root / "proj2_data"
output_dir = root / "assets"


# Image loading and display helpers

def read_image(filename, max_side=600):
    with Image.open(input_dir / filename) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64) / 255


def gray(image):
    return np.sum(image * [0.299, 0.587, 0.114], axis=-1)


def signed_view(image, scale=None):
    # Display negative and positive values around middle gray.
    if scale is None:
        scale = np.max(np.abs(image))
    return 0.5 + image / (2 * max(scale, 1e-12))


def save_image(filename, image):
    pixels = np.uint8(np.round(np.clip(image, 0, 1) * 255))
    image = Image.fromarray(pixels)
    path = output_dir / filename
    if filename.endswith(".png"):
        image.save(path)
    else:
        image.save(path, quality=92, subsampling=0)


def save_blend_process(name, blend, levels):
    # Use the same display scale for both sources and their combined band.
    last_level = len(blend["la"]) - 1
    for level in levels:
        scale = max(np.max(np.abs(blend["la"][level])),
                    np.max(np.abs(blend["lb"][level])))
        for key in ("ca", "cb", "blended"):
            image = blend[key][level]
            if level < last_level:
                image = signed_view(image, scale)
            save_image(f"{name}_{key}_{level}.jpg", image)
    for key in ("ca", "cb"):
        save_image(f"{name}_{key}_sum.jpg", blend[key].sum(axis=0))


def main():
    output_dir.mkdir(exist_ok=True)

    # Part 1.1: Convolutions from Scratch
    selfie = gray(read_image("selfie.jpeg", 256))
    save_image("selfie_gray.jpg", selfie)
    kernels = (("box", np.ones((9, 9)) / 81), ("dx", Dx), ("dy", Dy))
    methods = (("four loops", convolve_four_loops), ("two loops", convolve_two_loops),
               ("SciPy", convolve2d))
    timings = []
    for name, kernel in kernels:
        reference = convolve2d(selfie, kernel, mode="same")
        for method_name, method in methods:
            times = []
            for _ in range(3):
                start = perf_counter()
                result = method(selfie, kernel, mode="same")
                times.append(perf_counter() - start)
            elapsed = np.median(times)
            error = np.max(np.abs(result - reference))
            timings.append([name, method_name, elapsed, error])
            print(f"{name}, {method_name}: {elapsed:.4f}s, max error {error:.2e}")
            if method_name == "two loops":
                if name != "box":
                    result = signed_view(result)
                save_image(f"selfie_{name}.jpg", result)
    with (output_dir / "timings.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["filter", "method", "seconds", "max_error"])
        writer.writerows(timings)

    # Part 1.2: Finite Difference Operator
    camera = gray(read_image("cameraman.png"))
    dx, dy, magnitude = gradients(camera)
    save_image("camera.jpg", camera)
    save_image("camera_dx.jpg", signed_view(dx))
    save_image("camera_dy.jpg", signed_view(dy))
    save_image("camera_magnitude.jpg", magnitude)
    save_image("camera_magnitude_visible.jpg", magnitude * 4)
    for threshold in (0.08, 0.12, 0.18):
        save_image(f"camera_edges_{threshold:.2f}.png", magnitude > threshold)

    # Part 1.3: Derivative of Gaussian (DoG) Filter
    sequential, dog, dog_kernels = dog_gradients(camera, sigma=2)
    sequential_dx, sequential_dy = sequential
    dog_dx, dog_dy = dog
    dog_magnitude = np.hypot(dog_dx, dog_dy)
    save_image("camera_blur.jpg", gaussian_blur(camera, 2))
    save_image("dog_dx.jpg", signed_view(dog_dx))
    save_image("dog_dy.jpg", signed_view(dog_dy))
    save_image("sequential_magnitude.jpg", np.hypot(sequential_dx, sequential_dy))
    save_image("dog_magnitude.jpg", dog_magnitude)
    save_image("dog_magnitude_visible.jpg", dog_magnitude * 4)
    for threshold in (0.02, 0.035, 0.05):
        save_image(f"dog_edges_{threshold:.3f}.png", dog_magnitude > threshold)
    gaussian = gaussian_kernel(2)
    save_image("gaussian_kernel.png", gaussian / gaussian.max())
    for axis, kernel in zip(("x", "y"), dog_kernels):
        save_image(f"dog_kernel_{axis}.png", signed_view(kernel))
    error = np.max(np.abs(np.array(sequential) - np.array(dog)))
    print(f"DoG maximum difference: {error:.2e}")

    # Part 2.1: Image Sharpening
    for name, filename in (("taj", "taj.jpg"), ("building", "building.jpeg")):
        image = read_image(filename, 480)
        blurred = gaussian_blur(image, 2)
        save_image(f"{name}_original.jpg", image)
        save_image(f"{name}_blur.jpg", blurred)
        save_image(f"{name}_high.jpg", 0.5 + 3 * (image - blurred))
        for amount in (0.5, 1, 2, 4):
            save_image(f"{name}_sharp_{amount:g}.jpg", unsharp_mask(image, 2, amount))

    # Blur a sharp image, then see how much sharpening can recover.
    image = read_image("building.jpeg", 480)
    blurred = gaussian_blur(image, 2)
    restored = unsharp_mask(blurred, 2, 1.5)
    save_image("evaluation_original.jpg", image)
    save_image("evaluation_blurred.jpg", blurred)
    save_image("evaluation_restored.jpg", restored)
    blurred_error = np.mean((image - blurred) ** 2)
    restored_error = np.mean((image - np.clip(restored, 0, 1)) ** 2)
    print(f"Blurred MSE: {blurred_error:.6f}")
    print(f"Resharpened MSE: {restored_error:.6f}")

    # Part 2.2: Hybrid Images
    # Eye landmarks are fractions of each image's width and height.
    hybrid_gain = 1.25
    hybrids = [
        dict(name="derek_nutmeg", low="DerekPicture.jpg", high="nutmeg.jpg",
             low_points=((0.411, 0.337), (0.601, 0.325)),
             high_points=((0.423, 0.271), (0.534, 0.347)),
             target=((0.36, 0.40), (0.64, 0.40)), size=(400, 440), sigmas=(9, 4)),
        dict(name="headshot_miya", low="headshot.jpg", high="miya.jpeg",
             low_points=((0.463, 0.271), (0.544, 0.269)),
             high_points=((0.407, 0.358), (0.528, 0.341)),
             target=((0.36, 0.40), (0.64, 0.40)), size=(400, 440), sigmas=(9, 3)),
        dict(name="dog_cat", low="dog.jpg", high="cat.jpg",
             low_points=((0.29, 0.51), (0.684, 0.554)),
             high_points=((0.305, 0.533), (0.674, 0.54)),
             target=((0.29, 0.48), (0.71, 0.48)), size=(410, 361), sigmas=(6, 2)),
    ]
    for pair in hybrids:
        name = pair["name"]
        low_original = read_image(pair["low"], 800)
        high_original = read_image(pair["high"], 800)
        low_aligned = align_to_landmarks(low_original, pair["low_points"], pair["target"], pair["size"])
        high_aligned = align_to_landmarks(high_original, pair["high_points"], pair["target"], pair["size"])
        low_aligned = gray(low_aligned)
        high_aligned = gray(high_aligned)
        sigma_low, sigma_high = pair["sigmas"]
        low_filtered, high_filtered, hybrid = hybrid_image(
            low_aligned, high_aligned, sigma_low, sigma_high, hybrid_gain)

        save_image(f"{name}_original_low.jpg", low_original)
        save_image(f"{name}_original_high.jpg", high_original)
        save_image(f"{name}_aligned_low.jpg", low_aligned)
        save_image(f"{name}_aligned_high.jpg", high_aligned)
        save_image(f"{name}_hybrid.jpg", hybrid)
        small_hybrid = cv2.resize(np.clip(hybrid, 0, 1), None, fx=0.2, fy=0.2,
                                  interpolation=cv2.INTER_AREA)
        save_image(f"{name}_far.jpg", small_hybrid)

        # Show the filtering steps, Fourier spectra, and cutoff trials for this pair.
        if name == "derek_nutmeg":
            save_image(f"{name}_low.jpg", low_filtered)
            save_image(f"{name}_high.jpg", 0.5 + high_filtered)
            images = (low_aligned, high_aligned, low_filtered, high_filtered, hybrid)
            spectra = [log_spectrum(image) for image in images]
            scale = max(spectrum.max() for spectrum in spectra)
            labels = ("input_low", "input_high", "low", "high", "hybrid")
            for label, spectrum in zip(labels, spectra):
                save_image(f"{name}_fft_{label}.jpg", spectrum / scale)
            # Use the selected gain in every trial to compare only the cutoffs.
            for sigma_low, sigma_high in ((5, 2), (9, 4), (14, 6)):
                _, _, trial = hybrid_image(low_aligned, high_aligned, sigma_low, sigma_high, hybrid_gain)
                save_image(f"hybrid_trial_{sigma_low}_{sigma_high}.jpg", trial)

    # Part 2.3: Gaussian and Laplacian Stacks
    apple = read_image("apple.jpeg")
    orange = read_image("orange.jpeg")
    oraple_mask = np.zeros(apple.shape[:2])
    oraple_mask[:, :oraple_mask.shape[1] // 2] = 1
    oraple = multiresolution_blend(apple, orange, oraple_mask)

    # Gaussian and Laplacian levels for both fruit images.
    for key in ("ga", "gb"):
        for i, level in enumerate(oraple[key]):
            save_image(f"oraple_{key}_{i}.jpg", level)
    last_level = len(oraple["la"]) - 1
    for i in range(last_level + 1):
        scale = max(np.max(np.abs(oraple["la"][i])), np.max(np.abs(oraple["lb"][i])))
        for key in ("la", "lb"):
            level = oraple[key][i]
            if i < last_level:
                level = signed_view(level, scale)
            save_image(f"oraple_{key}_{i}.jpg", level)
    reconstruction_error = np.max(np.abs(oraple["la"].sum(axis=0) - apple))
    print(f"Stack reconstruction error: {reconstruction_error:.2e}")

    # Part 2.4: Multiresolution Blending
    save_image("oraple_apple.jpg", apple)
    save_image("oraple_orange.jpg", orange)
    save_image("oraple_mask.png", oraple_mask)
    hard_cut = oraple_mask[..., None] * apple + (1 - oraple_mask[..., None]) * orange
    save_image("oraple_hard.jpg", hard_cut)
    save_image("oraple_result.jpg", oraple["result"])
    for i, level in enumerate(oraple["gm"]):
        save_image(f"oraple_gm_{i}.png", level)
    save_blend_process("oraple", oraple, (0, 2, 4))

    # Trace the irregular rock opening on the 450 x 600 frame image.
    frame = read_image("frame.jpeg")
    height, width = frame.shape[:2]
    points = np.array([
        (180, 192), (203, 190), (216, 201), (236, 196), (268, 195),
        (278, 188), (291, 198), (322, 193), (328, 207), (336, 219),
        (338, 225), (328, 242), (322, 251), (330, 275), (331, 299),
        (327, 334), (328, 354), (324, 375), (324, 398), (329, 419),
        (325, 441), (303, 438), (284, 435), (262, 424), (243, 418),
        (224, 406), (205, 402), (184, 392), (180, 367), (180, 337),
        (176, 317), (174, 302), (167, 286), (170, 276), (160, 265),
        (151, 262), (155, 244), (156, 232), (154, 220), (166, 205), (174, 200),
    ])
    points = np.rint(points * [width / 450, height / 600]).astype(np.int32)
    window_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(window_mask, [points], 1)
    window_mask = window_mask.astype(float)
    x, y, w, h = cv2.boundingRect(points)
    plane = read_image("inside_frame.jpeg")
    positioned_plane = frame.copy()
    positioned_plane[y:y + h, x:x + w] = fit_image(plane, (w, h))
    save_image("blend_frame_plane.jpg", plane)

    # City2 is already upside down, creating the upper half of the mirror effect.
    city2 = fit_image(read_image("city2.jpeg"), (400, 600))
    city = fit_image(read_image("city.jpeg"), (400, 600))
    half_mask = np.zeros((600, 400))
    half_mask[:300] = 1
    blend_pairs = [("frame", positioned_plane, frame, window_mask, (1, 2, 4, 8)),
                   ("city_mirror", city2, city, half_mask, (2, 4, 8, 16))]
    for name, a, b, mask, sigmas in blend_pairs:
        blend = multiresolution_blend(a, b, mask, sigmas)
        masked_a = a * mask[..., None]
        masked_b = b * (1 - mask[..., None])
        save_image(f"blend_{name}_a.jpg", a)
        save_image(f"blend_{name}_b.jpg", b)
        save_image(f"blend_{name}_mask.png", mask)
        save_image(f"blend_{name}_masked_a.jpg", masked_a)
        save_image(f"blend_{name}_masked_b.jpg", masked_b)
        save_image(f"blend_{name}_hard.jpg", masked_a + masked_b)
        save_image(f"blend_{name}_result.jpg", blend["result"])
        if name == "frame":
            save_blend_process("blend_frame", blend, (0, 2, 4))


if __name__ == "__main__":
    main()
