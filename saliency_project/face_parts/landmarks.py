# face_parts/landmarks.py
"""
Facial landmark detection using MediaPipe Face Mesh.

This module is responsible ONLY for detecting facial landmarks
and returning them in pixel coordinates.

If no face is detected, returns None.
"""

import mediapipe as mp
import numpy as np
import cv2

mp_face_mesh = mp.solutions.face_mesh


class FaceLandmarkDetector:
    def __init__(self):
        # Static image mode is important for datasets
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def detect(self, image: np.ndarray):
        """
        Args:
            image: np.ndarray (H, W, 3), RGB or BGR

        Returns:
            dict with keys: left_eye, right_eye, nose, mouth
            Each value is a list of (x, y) pixel coordinates
            or None if no face is detected
        """
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None

        h, w, _ = image.shape
        landmarks = results.multi_face_landmarks[0].landmark

        def pts(indices):
            return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

        # MediaPipe landmark indices (stable & standard)
        left_eye = pts([33, 133, 160, 159, 158, 144, 145, 153])
        right_eye = pts([362, 263, 387, 386, 385, 373, 374, 380])
        nose = pts([1, 2, 98, 327, 168, 197])
        mouth = pts([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291])

        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "nose": nose,
            "mouth": mouth
        }