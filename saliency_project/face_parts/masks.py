# face_parts/masks.py
"""
Create and cache semantic face-part masks from landmarks.

Masks are computed ONCE per image and reused across all models.
"""

import torch
import numpy as np
import cv2
from pathlib import Path


def polygon_to_mask(image_shape, polygon):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return mask.astype(bool)


def build_face_masks(image, landmarks_dict):
    """
    Args:
        image: np.ndarray (H, W, 3)
        landmarks_dict: output of FaceLandmarkDetector.detect

    Returns:
        dict of boolean masks:
        eyes, nose, mouth, background
    """
    h, w, _ = image.shape

    left_eye = polygon_to_mask(image.shape, landmarks_dict["left_eye"])
    right_eye = polygon_to_mask(image.shape, landmarks_dict["right_eye"])
    eyes = left_eye | right_eye

    nose = polygon_to_mask(image.shape, landmarks_dict["nose"])
    mouth = polygon_to_mask(image.shape, landmarks_dict["mouth"])

    face = eyes | nose | mouth
    background = ~face

    return {
        "eyes": torch.from_numpy(eyes),
        "nose": torch.from_numpy(nose),
        "mouth": torch.from_numpy(mouth),
        "background": torch.from_numpy(background)
    }


def save_masks(masks, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(masks, path)


def load_masks(path: Path):
    return torch.load(path)