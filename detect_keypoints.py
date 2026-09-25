import cv2
import numpy as np


def detect_harris_keypoints(image, params: dict):
    """Detect interest points using the Harris corner detector.

    Returns an Nx2 array of (x, y) keypoint coordinates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    response = cv2.cornerHarris(
        gray,
        blockSize=params["block_size"],
        ksize=params["ksize"],
        k=params["k"],
    )

    threshold = params["threshold_ratio"] * response.max()
    ys, xs = np.where(response > threshold)
    keypoints = np.column_stack((xs, ys))

    return keypoints