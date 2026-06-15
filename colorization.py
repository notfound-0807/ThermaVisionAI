"""
colorization.py - Thermal Image Colorization Module
Applies OpenCV pseudo-color maps to grayscale thermal/infrared images
to produce visually interpretable false-color outputs.
"""

import cv2
import numpy as np


# Registry of supported colormaps: display label → OpenCV constant
COLORMAP_OPTIONS = {
    "Inferno": cv2.COLORMAP_INFERNO,
    "Jet": cv2.COLORMAP_JET,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Turbo": cv2.COLORMAP_TURBO,
}


def apply_colormap(gray_image: np.ndarray, colormap_name: str) -> np.ndarray:
    """
    Apply a pseudo-color map to a grayscale thermal image.

    Args:
        gray_image: Single-channel (grayscale) NumPy array, dtype uint8.
        colormap_name: Key from COLORMAP_OPTIONS dict (e.g. "Inferno").

    Returns:
        BGR color image as NumPy array with the selected colormap applied.

    Raises:
        ValueError: If colormap_name is not in COLORMAP_OPTIONS.
    """
    if colormap_name not in COLORMAP_OPTIONS:
        raise ValueError(
            f"Unknown colormap '{colormap_name}'. "
            f"Choose from: {list(COLORMAP_OPTIONS.keys())}"
        )

    # Ensure input is grayscale uint8
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    gray_image = gray_image.astype(np.uint8)

    # Apply the selected OpenCV colormap
    colormap_code = COLORMAP_OPTIONS[colormap_name]
    colorized = cv2.applyColorMap(gray_image, colormap_code)

    return colorized  # BGR format; convert to RGB for display in Streamlit


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image (OpenCV default) to RGB for Streamlit display.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def colorize_for_display(gray_image: np.ndarray, colormap_name: str) -> np.ndarray:
    """
    Full pipeline: apply colormap and convert to RGB for direct use with
    st.image() or PIL.

    Args:
        gray_image: Grayscale uint8 image.
        colormap_name: Colormap name string.

    Returns:
        RGB NumPy array ready for display or saving via PIL.
    """
    bgr_colorized = apply_colormap(gray_image, colormap_name)
    rgb_colorized = bgr_to_rgb(bgr_colorized)
    return rgb_colorized


def get_colormap_names() -> list:
    """Return the list of available colormap display names."""
    return list(COLORMAP_OPTIONS.keys())
