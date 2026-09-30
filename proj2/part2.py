import cv2
import numpy as np
from part1 import filter_image, gaussian_blur, gaussian_kernel


# Part 2.1: Image Sharpening

def unsharp_mask(image, sigma=2, amount=1):
    # Unsharp kernel: (1 + amount) * impulse - amount * Gaussian.
    kernel = -amount * gaussian_kernel(sigma)
    center = kernel.shape[0] // 2
    kernel[center, center] += 1 + amount
    return filter_image(image, kernel)


# Part 2.2: Hybrid Images

def hybrid_image(low_image, high_image, sigma_low, sigma_high, gain=1):
    low = gaussian_blur(low_image, sigma_low)
    high = high_image - gaussian_blur(high_image, sigma_high)
    # Keep the signed high frequencies; gain controls how visible they are.
    hybrid = low + gain * high
    return low, high, hybrid


def align_to_landmarks(image, points, target_points, size):
    # Points are fractions of image width/height; size is (width, height).
    # Complex division gives the rotation and scale that match both landmarks.
    h, w = image.shape[:2]
    source = np.asarray(points) * [w, h]
    target = np.asarray(target_points) * size
    src = complex(*(source[1] - source[0]))
    dst = complex(*(target[1] - target[0]))
    ratio = dst / src
    rotation = np.array([[ratio.real, -ratio.imag], [ratio.imag, ratio.real]])
    matrix = np.column_stack((rotation, target[0] - rotation @ source[0]))
    return cv2.warpAffine(image, matrix, tuple(size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def log_spectrum(image):
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    return np.log1p(np.abs(spectrum))


# Part 2.3: Gaussian and Laplacian Stacks

def gaussian_stack(image, sigmas=(2, 4, 8, 16, 32)):
    # Blur the previous level without downsampling.
    levels = [np.asarray(image, dtype=np.float64)]
    for sigma in sigmas:
        levels.append(gaussian_blur(levels[-1], sigma))
    return np.stack(levels)


def laplacian_stack(gaussians):
    bands = gaussians[:-1] - gaussians[1:]
    # Keep the final low-frequency level so summing the stack reconstructs the image.
    return np.concatenate((bands, gaussians[-1:]), axis=0)


# Part 2.4: Multiresolution Blending

def multiresolution_blend(a, b, mask, sigmas=(2, 4, 8, 16, 32)):
    ga = gaussian_stack(a, sigmas)
    gb = gaussian_stack(b, sigmas)
    gm = gaussian_stack(mask, sigmas)
    la = laplacian_stack(ga)
    lb = laplacian_stack(gb)
    # Share the 2D mask across color channels; white selects A, black selects B.
    weights = gm[..., None] if a.ndim == 3 else gm
    contribution_a = weights * la
    contribution_b = (1 - weights) * lb
    blended = contribution_a + contribution_b
    return dict(result=blended.sum(axis=0), ga=ga, gb=gb, gm=gm, la=la, lb=lb,
                ca=contribution_a, cb=contribution_b, blended=blended)


def fit_image(image, size):
    height, width = image.shape[:2]
    scale = max(size[0] / width, size[1] / height)
    resized = cv2.resize(image, (int(np.ceil(width * scale)), int(np.ceil(height * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    y = (resized.shape[0] - size[1]) // 2
    x = (resized.shape[1] - size[0]) // 2
    return resized[y:y + size[1], x:x + size[0]]
