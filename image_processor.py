"""
image_processor.py — Image loading, orientation normalization, perspective correction,
and CLAHE contrast enhancement for the Attendance Sheet Digitization System.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessedImage:
    """Container for all image representations needed across pipeline stages."""

    original_bgr: np.ndarray
    rectified_bgr: np.ndarray
    gray: np.ndarray
    enhanced_gray: np.ndarray
    binary_inv: np.ndarray
    ink_response: np.ndarray
    homography: Optional[np.ndarray] = None
    table_corners: Optional[np.ndarray] = None


def load_image(image_source: str | bytes | np.ndarray) -> np.ndarray:
    """Load a BGR image from a file path, raw bytes, or a NumPy array."""
    if isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:
            return cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
        if image_source.shape[2] == 4:
            return cv2.cvtColor(image_source, cv2.COLOR_RGBA2BGR)
        return image_source.copy()

    if isinstance(image_source, (bytes, bytearray)):
        arr = np.frombuffer(image_source, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode uploaded image bytes.")
        return image

    if isinstance(image_source, str):
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Could not read the image: {image_source}")
        image = cv2.imread(image_source, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read the image: {image_source}")
        return image

    raise TypeError(f"Unsupported image_source type: {type(image_source)}")


def normalize_orientation(image_bgr: np.ndarray) -> np.ndarray:
    """
    Ensure the attendance register is in portrait orientation with the table right-side up.
    If an image is accidentally rotated 90 degrees (landscape where horizontal lines are vertical),
    rotate it upright.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )
    h_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    )
    v_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    )
    # A register with ~25-35 student rows has higher horizontal line density than vertical
    if w > h * 1.15 and np.sum(v_lines > 0) > np.sum(h_lines > 0) * 1.4:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    return image_bgr


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_table_corners(binary_inv: np.ndarray) -> Optional[np.ndarray]:
    """
    Locate the four outer corners of the main attendance grid table using
    morphological horizontal and vertical line masks.
    """
    h_lines = cv2.morphologyEx(
        binary_inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    )
    v_lines = cv2.morphologyEx(
        binary_inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    )
    grid_mask = cv2.bitwise_or(h_lines, v_lines)
    grid_mask = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    img_area = binary_inv.shape[0] * binary_inv.shape[1]
    if cv2.contourArea(largest) < 0.15 * img_area:
        return None

    hull = cv2.convexHull(largest).reshape(-1, 2).astype(np.float32)
    return _order_quad_points(hull)


def preprocess_image(
    image_source: str | bytes | np.ndarray,
    output_save_path: Optional[str] = "output/preprocessed.jpg",
    apply_perspective_warp: bool = False,
) -> PreprocessedImage:
    """
    Step 2 — Load the image, normalize orientation, enhance contrast with CLAHE,
    compute local shadow-invariant ink response, and optionally save `output/preprocessed.jpg`.

    Note: By default `apply_perspective_warp=False` keeps the full page visible while
    `grid_detector.py` tracks exact tilted/curved horizontal and vertical line models
    directly across the page. Setting `apply_perspective_warp=True` warps the table
    corners to a flat rectangle.
    """
    raw_bgr = load_image(image_source)
    oriented_bgr = normalize_orientation(raw_bgr)

    gray0 = cv2.cvtColor(oriented_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced0 = clahe.apply(gray0)
    binary0 = cv2.adaptiveThreshold(
        enhanced0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )

    corners = detect_table_corners(binary0)
    homography = None
    rectified_bgr = oriented_bgr.copy()

    if apply_perspective_warp and corners is not None:
        tl, tr, br, bl = corners
        width_top = np.linalg.norm(tr - tl)
        width_bot = np.linalg.norm(br - bl)
        max_w = int(max(width_top, width_bot))

        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_h = int(max(height_left, height_right))

        margin_x, margin_y = 40, 60
        dst = np.array(
            [
                [margin_x, margin_y],
                [margin_x + max_w - 1, margin_y],
                [margin_x + max_w - 1, margin_y + max_h - 1],
                [margin_x, margin_y + max_h - 1],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(corners, dst)
        rectified_bgr = cv2.warpPerspective(
            oriented_bgr,
            homography,
            (max_w + 2 * margin_x, max_h + 2 * margin_y),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    binary_inv = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )

    # Local high-pass ink response removes paper crease shadows while keeping pen strokes
    local_bg = cv2.GaussianBlur(gray, (15, 15), 0).astype(np.float32)
    ink_response = np.clip(local_bg - gray.astype(np.float32), 0, 255)

    if output_save_path:
        os.makedirs(os.path.dirname(output_save_path) or ".", exist_ok=True)
        cv2.imwrite(output_save_path, enhanced)

    return PreprocessedImage(
        original_bgr=raw_bgr,
        rectified_bgr=rectified_bgr,
        gray=gray,
        enhanced_gray=enhanced,
        binary_inv=binary_inv,
        ink_response=ink_response,
        homography=homography,
        table_corners=corners,
    )
