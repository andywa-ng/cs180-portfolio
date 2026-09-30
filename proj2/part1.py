import cv2
import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d

# Part 1.1: Convolutions from Scratch


def pad_for_convolution(image, kernel, mode="same"):
    # Zero-pad for "same" or "full", then flip the kernel for convolution.
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    if mode == "same":
        # Put the extra pixel on top/left for even kernels to match SciPy.
        padding = ((kh // 2, (kh - 1) // 2), (kw // 2, (kw - 1) // 2))
    else:
        padding = ((kh - 1, kh - 1), (kw - 1, kw - 1))
    return np.pad(image, padding, mode="constant"), kernel[::-1, ::-1]


def convolve_four_loops(image, kernel, mode="same"):
    padded, flipped = pad_for_convolution(image, kernel, mode)
    kh, kw = flipped.shape
    h, w = padded.shape[0] - kh + 1, padded.shape[1] - kw + 1
    result = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            for ky in range(kh):
                for kx in range(kw):
                    result[y, x] += padded[y + ky, x + kx] * flipped[ky, kx]
    return result


def convolve_two_loops(image, kernel, mode="same"):
    padded, flipped = pad_for_convolution(image, kernel, mode)
    kh, kw = flipped.shape
    h, w = padded.shape[0] - kh + 1, padded.shape[1] - kw + 1
    result = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            result[y, x] = np.sum(padded[y:y + kh, x:x + kw] * flipped)
    return result


# Part 1.2: Finite Difference Operator

Dx = np.array([[1.0, -1.0]])
Dy = Dx.T


def filter_image(image, kernel):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return convolve2d(image, kernel, mode="same", boundary="symm")
    # For color images, filter each channel independently.
    return np.stack([convolve2d(image[..., c], kernel, mode="same", boundary="symm")
                     for c in range(image.shape[2])], axis=-1)


def gradients(image):
    dx = filter_image(image, Dx)
    dy = filter_image(image, Dy)
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    return dx, dy, magnitude


# Part 1.3: Derivative of Gaussian (DoG) Filter

def gaussian_kernel(sigma, size=None):
    if size is None:
        size = 2 * int(np.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(size, sigma, cv2.CV_64F)
    # Outer product of the 1D Gaussian gives the 2D kernel.
    return g @ g.T


def gaussian_blur(image, sigma):
    image = np.asarray(image, dtype=np.float64)
    size = 2 * int(np.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(size, sigma, cv2.CV_64F).ravel()
    # Apply the Gaussian along rows and columns without mixing color channels.
    blurred = convolve1d(image, g, axis=0, mode="reflect")
    return convolve1d(blurred, g, axis=1, mode="reflect")


def dog_gradients(image, sigma=2):
    # Extend once so both methods use the same pixels at the image boundaries.
    gaussian = gaussian_kernel(sigma)
    sequential, direct, kernels = [], [], []
    for derivative in (Dx, Dy):
        # Combine smoothing and differentiation into a single DoG kernel.
        kernel = convolve2d(gaussian, derivative, mode="full")
        kh, kw = kernel.shape
        padded = np.pad(image, ((kh // 2, (kh - 1) // 2),
                                (kw // 2, (kw - 1) // 2)), mode="symmetric")
        blurred = convolve2d(padded, gaussian, mode="valid")
        sequential.append(convolve2d(blurred, derivative, mode="valid"))
        direct.append(convolve2d(padded, kernel, mode="valid"))
        kernels.append(kernel)
    return sequential, direct, kernels
