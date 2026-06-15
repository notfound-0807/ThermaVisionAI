"""
enhancement.py - Infrared Image Enhancement Pipeline
Implements denoising, CLAHE, and sharpening for thermal images.
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if it's in color (BGR/RGB).
    Returns the image unchanged if already single-channel.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Handle RGBA images
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image  # Already grayscale


def denoise(gray: np.ndarray) -> np.ndarray:
    """
    Apply Non-Local Means Denoising to reduce thermal sensor noise.
    h=10 controls filter strength; higher removes more noise but loses detail.
    templateWindowSize=7: size of the patch used to compare pixels.
    searchWindowSize=21: size of the window to search for similar patches.
    """
    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )
    return denoised


def apply_clahe(denoised: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances local contrast without over-amplifying noise.
    clipLimit=2.0: prevents over-amplification of noise in uniform regions.
    tileGridSize=(8, 8): divides image into 8x8 tiles for local equalization.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced


def sharpen(enhanced: np.ndarray) -> np.ndarray:
    """
    Apply edge sharpening using a custom convolution kernel.
    The kernel emphasizes the center pixel and subtracts its neighbors,
    effectively enhancing edges and fine details in the thermal image.
    """
    sharpening_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(enhanced, ddepth=-1, kernel=sharpening_kernel)
    # Clip values to valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


def run_pipeline(image: np.ndarray) -> dict:
    """
    Run the full enhancement pipeline and return all intermediate results.

    Args:
        image: Input image as a NumPy array (BGR or grayscale).

    Returns:
        Dictionary with keys:
            'gray'     – grayscale version of input
            'denoised' – after Non-Local Means denoising
            'clahe'    – after CLAHE contrast enhancement
            'sharpened'– after edge sharpening (final enhanced output)
    """
    gray = to_grayscale(image)
    denoised = denoise(gray)
    clahe_result = apply_clahe(denoised)
    sharpened = sharpen(clahe_result)

    return {
        "gray": gray,
        "denoised": denoised,
        "clahe": clahe_result,
        "sharpened": sharpened,
    }


def compute_metrics(original_gray: np.ndarray, enhanced: np.ndarray) -> dict:
    """
    Compute quality metrics comparing original and enhanced images.

    Args:
        original_gray: Grayscale original image.
        enhanced: Final enhanced (sharpened) image.

    Returns:
        Dictionary with metric values for display.
    """
    # Mean pixel intensity of enhanced image
    mean_intensity = float(np.mean(enhanced))

    # Contrast = standard deviation of pixel values
    original_contrast = float(np.std(original_gray))
    enhanced_contrast = float(np.std(enhanced))

    if original_contrast > 0:
        contrast_improvement = ((enhanced_contrast - original_contrast) / original_contrast) * 100
    else:
        contrast_improvement = 0.0

    # Image resolution string
    h, w = enhanced.shape[:2]
    resolution = f"{w} × {h} px"

    return {
        "mean_intensity": round(mean_intensity, 2),
        "contrast_improvement": round(contrast_improvement, 2),
        "resolution": resolution,
        "original_contrast": round(original_contrast, 2),
        "enhanced_contrast": round(enhanced_contrast, 2),
    }
