# face_parts/landmarks.py

"""
Facial landmark detection using the new MediaPipe Tasks API.

Compatible with MediaPipe >= 0.10.
"""

import numpy as np
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceLandmarkDetector:
    def __init__(self):
        # Load pre-trained face landmark model
        base_options = python.BaseOptions(
        model_asset_path="saliency_project/face_parts/face_landmarker.task"
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray):
        """
        Args:
            image: np.ndarray (H, W, 3), BGR or RGB

        Returns:
            dict with:
                left_eye, right_eye, nose, mouth
            or None if no face detected
        """

        # Convert to RGB (MediaPipe expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w, _ = image.shape

        def pts(indices):
            return [
                (int(landmarks[i].x * w),
                 int(landmarks[i].y * h))
                for i in indices
            ]

        # Same semantic groups as before
        left_eye = pts([33, 133, 160, 159, 158, 144, 145, 153])
        right_eye = pts([362, 263, 387, 386, 385, 373, 374, 380])
        nose = pts([1,    # nose tip
            2,    # upper bridge
            98,   # left bridge
            327,  # right bridge
            168,  # center root
            195,  # left nostril top
            5,    # left nostril side
            4,    # bottom tip
            275,  # right nostril side
            440,  # right nostril top
            ])
        mouth = pts([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291])

        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "nose": nose,
            "mouth": mouth
        }