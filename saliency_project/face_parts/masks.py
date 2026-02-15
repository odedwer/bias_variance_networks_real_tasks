# face_parts/masks.py
"""
Create and cache semantic face-part masks from landmarks.

Masks are computed ONCE per image and reused across all models.
"""

import torch
import numpy as np
import cv2
from pathlib import Path


def polygon_to_mask(points, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def dilate_mask(mask, pixels=2):
    """
    Adds a margin around a binary mask using morphological dilation.
    """
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
    return cv2.dilate(mask, kernel)



def build_face_masks(image, landmarks, margin_pixels=2):
    """
    image: np.ndarray (H,W,3)
    landmarks: dict of facial landmarks
    """

    h, w = image.shape[:2]

    left_eye = polygon_to_mask(landmarks["left_eye"], h, w)
    right_eye = polygon_to_mask(landmarks["right_eye"], h, w)
    eyes = np.clip(left_eye + right_eye, 0, 1)

    nose = polygon_to_mask(landmarks["nose"], h, w)
    mouth = polygon_to_mask(landmarks["mouth"], h, w)

    # --- add margin ---
    eyes = dilate_mask(eyes, margin_pixels)
    nose = dilate_mask(nose, margin_pixels)
    mouth = dilate_mask(mouth, margin_pixels)

    # --- ensure no overlap ---
    eyes = eyes.astype(bool)
    nose = np.logical_and(nose, ~eyes)
    mouth = np.logical_and(mouth, ~(eyes | nose))

    background = ~(eyes | nose | mouth)

    return {
        "eyes": torch.from_numpy(eyes),
        "nose": torch.from_numpy(nose),
        "mouth": torch.from_numpy(mouth),
        "background": torch.from_numpy(background),
    }



def save_masks(masks, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(masks, path)


def load_masks(path: Path):
    return torch.load(path)