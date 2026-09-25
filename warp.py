import cv2
import numpy as np

_INTERPOLATION_MAP = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}


def warp_to_rectangle(image, corners, params: dict):
    """Correct perspective: map the 4 detected corners onto a flat rectangle.

    corners must be ordered: top-left, top-right, bottom-right, bottom-left.
    Returns the warped (flattened) image.
    """
    width = params["output_width"]
    height = params["output_height"]
    interpolation = _INTERPOLATION_MAP[params["interpolation"]]

    src_points = corners.astype("float32")
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype="float32")

    H = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, H, (width, height), flags=interpolation)

    return warped