from datetime import datetime

import cv2
import yaml

from capture import get_frame
from detect_keypoints import detect_harris_keypoints
from find_corners import find_page_corners
from warp import warp_to_rectangle
from save import save_snapshot


def draw_points(image, points, color=(0, 255, 0), radius=3):
    """Return a copy of image with points drawn on it."""
    out = image.copy()
    for x, y in points:
        cv2.circle(out, (int(x), int(y)), radius, color, -1)
    return out


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    run_id = datetime.now().strftime("%H-%M-%S")

    # Stage 1: capture
    frame = get_frame(cfg["capture"])
    save_snapshot(frame, "00_captured", run_id)

    # Stage 2: Harris keypoints
    keypoints = detect_harris_keypoints(frame, cfg["harris"])
    keypoints_preview = draw_points(frame, keypoints, color=(0, 255, 0), radius=1)
    save_snapshot(keypoints_preview, "01_harris_keypoints", run_id)

    # Stage 3: page corners (contour + keypoint refinement)
    corners = find_page_corners(frame, keypoints, cfg["corner_selection"])
    if corners is None:
        raise RuntimeError("No page detected — check config parameters (epsilon, min area)")

    corners_preview = draw_points(frame, corners, color=(0, 0, 255), radius=6)
    save_snapshot(corners_preview, "02_corners_selected", run_id)

    # Stage 4: perspective correction
    result = warp_to_rectangle(frame, corners, cfg["warp"])
    save_snapshot(result, "03_warped_output", run_id)

    cv2.imshow("Corrected document", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()