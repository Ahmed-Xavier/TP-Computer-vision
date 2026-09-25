import cv2
import numpy as np
from logger import get_logger

logger = get_logger(__name__)


def _order_corners(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]       # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]       # bottom-right: largest x+y
    ordered[1] = pts[np.argmin(diff)]    # top-right: smallest y-x
    ordered[3] = pts[np.argmax(diff)]    # bottom-left: largest y-x
    return ordered


def _snap_to_nearest_keypoint(corner, keypoints, radius):
    """Refine a contour corner by snapping to the nearest Harris keypoint within radius."""
    if keypoints is None or len(keypoints) == 0:
        return corner

    distances = np.linalg.norm(keypoints - corner, axis=1)
    idx = np.argmin(distances)

    if distances[idx] <= radius:
        return keypoints[idx].astype("float32")
    return corner


def find_page_corners(image, keypoints, params: dict):
    """Find the 4 corners of the page using a contour, refined with nearby Harris keypoints.

    Returns a 4x2 array of ordered corners (top-left, top-right, bottom-right, bottom-left),
    or None if no suitable contour is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = image.shape[0] * image.shape[1]
    min_area = params["min_contour_area_ratio"] * frame_area

    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not candidates:
        return None

    largest = max(candidates, key=cv2.contourArea)

    perimeter = cv2.arcLength(largest, True)
    epsilon = params["approx_poly_epsilon"] * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) != 4:
        logger.error(
            "No quadrilateral: approx_vertices=%d, contour_area=%.1f",
            len(approx),
            cv2.contourArea(largest),
        )
        return None

    corners = approx.reshape(4, 2)
    corners = _order_corners(corners)

    radius = params["keypoint_search_radius"]
    refined = np.array([
        _snap_to_nearest_keypoint(corner, keypoints, radius)
        for corner in corners
    ])

    return refined