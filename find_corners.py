from datetime import datetime
import cv2
import numpy as np
from logger import get_logger
from save import save_snapshot

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
    h, w = image.shape[:2]
    frame_area = h * w
    min_area = params["min_contour_area_ratio"] * frame_area
    num_keypoints = len(keypoints) if keypoints is not None else 0

    run_id = datetime.now().strftime("%H-%M-%S")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    save_snapshot(edges, "01b_canny", run_id)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_preview = image.copy()
    if len(contours_preview.shape) == 2:
        contours_preview = cv2.cvtColor(contours_preview, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contours_preview, contours, -1, (0, 255, 0), 2)
    save_snapshot(contours_preview, "01c_contours", run_id)

    total_contours = len(contours) if contours else 0

    if not contours:
        logger.error(
            "Page detection failed: No contours found. "
            "[total_contours=%d, frame_w=%d, frame_h=%d, frame_area=%d, min_required_area=%.1f, harris_keypoints=%d]",
            total_contours,
            w,
            h,
            frame_area,
            min_area,
            num_keypoints,
        )
        return None

    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    num_candidates = len(candidates)

    if not candidates:
        largest_found_area = max([cv2.contourArea(c) for c in contours]) if contours else 0.0
        largest_found_ratio = largest_found_area / frame_area if frame_area > 0 else 0.0
        logger.error(
            "Page detection failed: No contours pass minimum area threshold. "
            "[total_contours=%d, frame_w=%d, frame_h=%d, frame_area=%d, "
            "min_required_area=%.1f, candidates=%d, largest_contour_area=%.1f, "
            "largest_area_ratio=%.4f, harris_keypoints=%d]",
            total_contours,
            w,
            h,
            frame_area,
            min_area,
            num_candidates,
            largest_found_area,
            largest_found_ratio,
            num_keypoints,
        )
        return None

    largest = max(candidates, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)
    largest_area_ratio = largest_area / frame_area if frame_area > 0 else 0.0

    perimeter = cv2.arcLength(largest, True)
    epsilon = params["approx_poly_epsilon"] * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)
    num_vertices = len(approx)

    if num_vertices != 4:
        logger.error(
            "Page detection failed: Largest candidate does not produce exactly 4 vertices. "
            "[total_contours=%d, frame_w=%d, frame_h=%d, frame_area=%d, "
            "min_required_area=%.1f, candidates=%d, largest_candidate_area=%.1f, "
            "largest_candidate_area_ratio=%.4f, epsilon=%.2f, approx_vertices=%d, "
            "harris_keypoints=%d]",
            total_contours,
            w,
            h,
            frame_area,
            min_area,
            num_candidates,
            largest_area,
            largest_area_ratio,
            epsilon,
            num_vertices,
            num_keypoints,
        )
        return None

    corners = approx.reshape(4, 2)
    corners = _order_corners(corners)

    radius = params["keypoint_search_radius"]
    refined = np.array([
        _snap_to_nearest_keypoint(corner, keypoints, radius)
        for corner in corners
    ])

    logger.error(
        "Page detection succeeded. "
        "[total_contours=%d, frame_w=%d, frame_h=%d, frame_area=%d, "
        "min_required_area=%.1f, candidates=%d, largest_candidate_area=%.1f, "
        "largest_candidate_area_ratio=%.4f, epsilon=%.2f, approx_vertices=%d, "
        "harris_keypoints=%d, corners=%s]",
        total_contours,
        w,
        h,
        frame_area,
        min_area,
        num_candidates,
        largest_area,
        largest_area_ratio,
        epsilon,
        num_vertices,
        num_keypoints,
        refined.tolist(),
    )

    return refined