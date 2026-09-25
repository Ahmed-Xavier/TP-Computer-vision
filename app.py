from datetime import datetime

import cv2
import yaml

from capture import get_frame
from detect_keypoints import detect_sift_features
from find_corners import find_page_corners
from warp import warp_to_rectangle
from save import save_snapshot


def draw_corners(image, corners):
    out = image.copy()

    for i, (x, y) in enumerate(corners):
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(
            out,
            str(i + 1),
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return out


def match_sift_features(original, warped, original_kp, warped_kp,
                        original_desc, warped_desc):

    if original_desc is None or warped_desc is None:
        return None

    if len(original_desc) < 2 or len(warped_desc) < 2:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    knn_matches = matcher.knnMatch(
        original_desc,
        warped_desc,
        k=2
    )

    good_matches = []

    for pair in knn_matches:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    match_image = cv2.drawMatches(
        original,
        original_kp,
        warped,
        warped_kp,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return match_image


def main():

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    run_id = datetime.now().strftime("%H-%M-%S")

    # --------------------------------------------------
    # 1. Capture
    # --------------------------------------------------

    frame = get_frame(cfg["capture"])
    save_snapshot(frame, "00_captured", run_id)

    # --------------------------------------------------
    # 2. SIFT on original image
    # --------------------------------------------------

    keypoints, descriptors = detect_sift_features(
        frame,
        cfg["sift"]
    )

    keypoints_preview = cv2.drawKeypoints(
        frame,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    save_snapshot(
        keypoints_preview,
        "01_sift_keypoints_original",
        run_id
    )

    print(f"SIFT keypoints in original: {len(keypoints)}")

    # --------------------------------------------------
    # 3. Get exactly 4 paper corners
    # --------------------------------------------------

    corners = find_page_corners(
    frame,
    cfg["corner_selection"]
    )

    corners_preview = draw_corners(frame, corners)

    save_snapshot(
        corners_preview,
        "02_corners_selected",
        run_id
    )

    print("Paper corners:")
    for i, corner in enumerate(corners):
        print(f"  {i + 1}: {corner}")

    # --------------------------------------------------
    # 4. Perspective transform
    # --------------------------------------------------
    # IMPORTANT:
    # Homography comes ONLY from the 4 paper corners.
    # SIFT is NOT used to calculate the homography.

    result = warp_to_rectangle(
        frame,
        corners,
        cfg["warp"]
    )

    save_snapshot(
        result,
        "03_warped_output",
        run_id
    )

    # --------------------------------------------------
    # 5. SIFT again on the warped result
    # --------------------------------------------------

    warped_keypoints, warped_descriptors = detect_sift_features(
        result,
        cfg["sift"]
    )

    warped_preview = cv2.drawKeypoints(
        result,
        warped_keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    save_snapshot(
        warped_preview,
        "04_sift_keypoints_result",
        run_id
    )

    print(f"SIFT keypoints in result: {len(warped_keypoints)}")

    # --------------------------------------------------
    # 6. Match original ↔ result
    # --------------------------------------------------

    match_image = match_sift_features(
        frame,
        result,
        keypoints,
        warped_keypoints,
        descriptors,
        warped_descriptors,
    )

    if match_image is not None:
        save_snapshot(
            match_image,
            "05_sift_correspondences",
            run_id
        )

        cv2.imshow(
            "SIFT correspondences",
            match_image
        )

    # --------------------------------------------------
    # 7. Show final result
    # --------------------------------------------------

    cv2.imshow(
        "Corrected document",
        result
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()